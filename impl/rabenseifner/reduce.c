/*
 * An implementation of Rabenseifner's reduce algorithm [1, 2].
 *
 * This algorithm is a combination of a reduce-scatter implemented with
 * recursive vector halving and recursive distance doubling, followed either
 * by a binomial tree gather [1].
 * 
 * Step 1. If the number of processes is not a power of two, reduce it to
 * the nearest lower power of two (p' = 2^{\lfloor\log_2 p\rfloor})
 * by removing r = p - p' extra processes as follows. In the first 2r processes
 * (ranks 0 to 2r - 1), all the even ranks send the second half of the input
 * vector to their right neighbor (rank + 1), and all the odd ranks send
 * the first half of the input vector to their left neighbor (rank тИТ 1).
 * The even ranks compute the reduction on the first half of the vector and
 * the odd ranks compute the reduction on the second half. The odd ranks then
 * send the result to their left neighbors (the even ranks). As a result,
 * the even ranks among the first 2r processes now contain the reduction with
 * the input vector on their right neighbors (the odd ranks). These odd ranks
 * do not participate in the rest of the algorithm, which leaves behind
 * a power-of-two number of processes. The first r even-ranked processes and
 * the last p - 2r processes are now renumbered from 0 to p' - 1.
 *
 * Step 2. The remaining processes now perform a reduce-scatter by using
 * recursive vector halving and recursive distance doubling. The even-ranked
 * processes send the second half of their buffer to rank + 1 and the odd-ranked
 * processes send the first half of their buffer to rank тИТ 1. All processes
 * then compute the reduction between the local buffer and the received buffer.
 * In the next log_2(p') - 1 steps, the buffers are recursively halved, and the
 * distance is doubled. At the end, each of the p' processes has 1/p' of the
 * total reduction result.
 *
 * Step 3. A binomial tree gather is performed by using recursive vector
 * doubling and distance halving. In the non-power-of-two case, if the root
 * happens to be one of those odd-ranked processes that would normally
 * be removed in the first step, then the role of this process and process 0
 * are interchanged.
 * 
 * Limitations: commutative operations only, count >= 2^{\lfloor\log_2 p\rfloor}
 * Recommendations: root = 0, otherwise it is required additional steps
 *                  in the root process.
 *
 * Memory consumption (per process):
 * 1) rank != root: 2 * count * typesize + 4 * log2(p) * sizeof(int) = O(count)
 * 2) rank == root: count * typesize + 4 * log2(p) * sizeof(int) = O(count)
 *
 * [1] Rajeev Thakur, Rolf Rabenseifner and William Gropp.
 *     Optimization of Collective Communication Operations in MPICH //
 *     The Int. Journal of High Performance Computing Applications. Vol 19,
 *     Issue 1, pp. 49--66.
 * [2] http://www.hlrs.de/mpi/myreduce.html.
 */
#undef FUNCNAME
#define FUNCNAME MPIR_Reduce_redscat_gather
#undef FCNAME
#define FCNAME MPL_QUOTE(FUNCNAME)
static int MPIR_Reduce_redscat_gather(
    const void *sendbuf,
    void *recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPID_Comm *comm_ptr,
    MPIR_Errflag_t *errflag )
{
    int mpi_errno = MPI_SUCCESS;
    int mpi_errno_ret = MPI_SUCCESS;
    int comm_size, rank, type_size ATTRIBUTE((unused)), pof2, rem, newrank;
    int mask, i, j, newdst, dst, nsteps, step, wsize;
    int newroot, newdst_tree_root, newroot_tree_root;
    MPI_Aint true_lb, true_extent, extent; 
    void *tmp_buf;
    int *rindex, *rcount, *sindex, *scount, count_lhalf, count_rhalf;

    MPIU_CHKLMEM_DECL(6);
    MPID_THREADPRIV_DECL;

    comm_size = comm_ptr->local_size;
    rank = comm_ptr->rank;

    /* Set op_errno to 0. Stored in perthread structure */
    MPID_THREADPRIV_GET;
    MPID_THREADPRIV_FIELD(op_errno) = 0;

    /* Create a temporary buffer */
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    MPID_Datatype_get_extent_macro(datatype, extent);

    /* I think this is the worse case, so we can avoid an assert()
     * inside the for loop should be buf+{this}?
     */
    MPIU_Ensure_Aint_fits_in_pointer(count * MPIR_MAX(extent, true_extent));

    MPIU_CHKLMEM_MALLOC(tmp_buf, void *, count*(MPIR_MAX(extent, true_extent)),
                        mpi_errno, "temporary buffer");
    /* Adjust for potential negative lower bound in datatype */
    tmp_buf = (void *)((char*)tmp_buf - true_lb);

    /* If I'm not the root, then my recvbuf may not be valid, therefore
     * I have to allocate a temporary one */
    if (rank != root) {
        MPIU_CHKLMEM_MALLOC(recvbuf, void *, 
                            count * (MPIR_MAX(extent, true_extent)),
                            mpi_errno, "receive buffer");
        recvbuf = (void *)((char*)recvbuf - true_lb);
    }

    if ((rank != root) || (sendbuf != MPI_IN_PLACE)) {
        mpi_errno = MPIR_Localcopy(sendbuf, count, datatype, recvbuf,
                                   count, datatype);
        if (mpi_errno) { MPIR_ERR_POP(mpi_errno); }
    }

    MPID_Datatype_get_size_macro(datatype, type_size);

    /*
     * Step 1. Reduce the number of processes to the nearest lower power of two
     * (p' = 2^{\lfloor\log_2 p\rfloor}) by removing r = p - p' processes.
     * 1. In the first 2r processes (ranks 0 to 2r - 1), all the even ranks send
     *    the second half of the input vector to their right neighbor (rank + 1)
     *    and all the odd ranks send the first half of the input vector to their
     *    left neighbor (rank тИТ 1).
     * 2. All 2r processes compute the reduction on their half.
     * 3. The odd ranks then send the result to their left neighbors
     *    (the even ranks).
     *
     * The even ranks (0 to 2r - 1) now contain the reduction with the input
     * vector on their right neighbors (the odd ranks). The first r even
     * processes and the p - 2r last processes are renumbered from
     * 0 to 2^{\floor(log_2 p)} - 1. These odd ranks do not participate in the
     * rest of the algorithm.
     */

    /* Find nearest power-of-two less than or equal to comm_size */
    pof2 = 1;
    nsteps = -1;
    while (pof2 <= comm_size) {  /* O(log(p)), FIXME: use flp2 and ilog2 */
        pof2 <<= 1;
        nsteps++;
    }        
    pof2 >>= 1;

    rem = comm_size - pof2;
    if (rank < 2 * rem) {
        count_lhalf = count / 2;
        count_rhalf = count - count_lhalf;

        if (rank % 2 != 0) { /* odd process -- exchange with rank - 1 */
            /*
             * Send the left half of the input vector to the left neighbor,
             * Recv the right half of the input vector from the left neighbor
             */
            mpi_errno = MPIC_Sendrecv(recvbuf, count_lhalf, datatype,
                                      rank - 1, MPIR_REDUCE_TAG,
                                      (char *)tmp_buf + count_lhalf * extent,
                                      count_rhalf, datatype, rank - 1,
                                      MPIR_REDUCE_TAG, comm_ptr,
                                      MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* Reduce on the right half of the buffers (result in recvbuf) */
            mpi_errno = MPIR_Reduce_local_impl((char *)tmp_buf +
                                               count_lhalf * extent,
                                               (char *)recvbuf +
                                               count_lhalf * extent,
                                               count_rhalf, datatype, op);
            
            /* Send the right half to the left neighbor */
            mpi_errno = MPIC_Send((char *)recvbuf + count_lhalf * extent,
                                  count_rhalf, datatype, rank - 1,
                                  MPIR_REDUCE_TAG, comm_ptr, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            /* Temporarily set the rank to -1 so that this process does not
               pariticipate in recursive doubling */
            newrank = -1;
            
        } else { /* even process -- exchange with rank + 1 */
            /*
             * Send the right half of the input vector to the right neighbor,
             * Recv the left half of the input vector from the right neighbor
             */
            mpi_errno = MPIC_Sendrecv((char *)recvbuf + count_lhalf * extent,
                                      count_rhalf, datatype, rank + 1,
                                      MPIR_REDUCE_TAG, tmp_buf, count_lhalf,
                                      datatype, rank + 1, MPIR_REDUCE_TAG,
                                      comm_ptr, MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* Reduce on the left half of the buffers (result in recvbuf) */
            mpi_errno = MPIR_Reduce_local_impl(tmp_buf, recvbuf, count_lhalf,
                                               datatype, op);

            /* Recv the right half from the right neighbor */
            mpi_errno = MPIC_Recv((char *)recvbuf + count_lhalf * extent,
                                  count_rhalf, datatype, rank + 1,
                                  MPIR_REDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE,
                                  errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }
            newrank = rank / 2;
        }
    } else { /* rank >= 2 * rem */
        newrank = rank - rem;
    }

    /*
     * Step 2. Reduce-scatter implemented with recursive vector halving and
     * recursive distance doubling. We have p' = 2^{\lfloor\log_2 p\rfloor}
     * power-of-two number of processes with new ranks and result in recvbuf.
     * 
     * The even-ranked processes send the right half of their buffer to rank + 1
     * and the odd-ranked processes send the left half of their buffer to
     * rank - 1. All processes then compute the reduction between the local
     * buffer and the received buffer. In the next \log_2(p') - 1 steps, the
     * buffers are recursively halved, and the distance is doubled. At the end,
     * each of the p' processes has 1 / p' of the total reduction result.
     */
    MPIU_CHKLMEM_MALLOC(rindex, int *, nsteps * sizeof(*rindex), mpi_errno,
                        "rindex buffer");
    MPIU_CHKLMEM_MALLOC(rcount, int *, nsteps * sizeof(*rcount), mpi_errno,
                        "rcount buffer");
    MPIU_CHKLMEM_MALLOC(sindex, int *, nsteps * sizeof(*sindex), mpi_errno,
                        "sindex buffer");
    MPIU_CHKLMEM_MALLOC(scount, int *, nsteps * sizeof(*scount), mpi_errno,
                        "scount buffer");

    if (newrank != -1) {
        step = 0;
        wsize = count; 
        sindex[0] = rindex[0] = 0;

        for (mask = 1; mask < pof2; mask <<= 1) {
            /*
             * On each iteration: rindex[step] = sindex[step] -- begining of the
             * current window. Length of the current window is storded in wsize.
             */
            newdst = newrank ^ mask;
            /* Find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 : newdst + rem;

            if (rank < dst) {
                /* Recv into the left half of the current window, send the right
                 * half of the window to the peer (perform reduce on the left
                 * half of the current window)
                 */
                rcount[step] = wsize / 2;
                scount[step] = wsize - rcount[step];
                sindex[step] = rindex[step] + rcount[step];
            } else {
                /* Recv into the right half of the current window, send the left
                 * half of the window to the peer (perform reduce on the right
                 * half of the current window)
                 */
                scount[step] = wsize / 2;
                rcount[step] = wsize - scount[step];
                rindex[step] = sindex[step] + scount[step];
            }

            /* Send part of data from the recvbuf, recv into the tmp_buf */
            mpi_errno = MPIC_Sendrecv((char *)recvbuf + sindex[step] * extent,
                                      scount[step], datatype, dst,
                                      MPIR_REDUCE_TAG,
                                      (char *)tmp_buf + rindex[step] * extent,
                                      rcount[step], datatype, dst,
                                      MPIR_REDUCE_TAG, comm_ptr,
                                      MPI_STATUS_IGNORE, errflag);
            if (mpi_errno) {
                /* for communication errors, just record the error but continue */
                *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
            }

            /* Local reduce: recvbuf[] = tmp_buf[] <op> recvbuf[] */
            mpi_errno = MPIR_Reduce_local_impl((char *)tmp_buf +
                                               rindex[step] * extent,
                                               (char *)recvbuf +
                                               rindex[step] * extent,
                                               rcount[step], datatype, op);

            /* Move the current window to the received message */
            rindex[step + 1] = rindex[step];
            sindex[step + 1] = rindex[step];
            wsize = rcount[step];
            step++;
        }
    }
    /*
     * Assertion: each process has 1 / p' of the total reduction result: 
     *   rcount[nsteps - 1] elements in the recvbuf[rindex[nsteps - 1]...].
     */

    /*
     * Setup the root process for gather operation.
     * Case 1: root < 2r and root is odd -- root process was excluded on step 1
     *         Recv data from process 0, newroot = 0, newrank = 0
     * Case 2: root < 2r and root is even: newroot = root / 2
     * Case 3: root >= 2r: newroot = root - r
     */
    newroot = 0;
    if (root < 2 * rem) {
        if (root % 2 != 0) {
            newroot = 0;
            if (rank == root) {
                /* Case 1: root < 2r and root is odd -- root process was
                 * excluded on step 1 (newrank == -1).
                 * Recv a data from the process 0.
                 */
                rindex[0] = 0;
                step = 0, wsize = count;
                for (mask = 1; mask < pof2; mask *= 2) { 
                    rcount[step] = wsize / 2;
                    scount[step] = wsize - rcount[step];
                    rindex[step] = 0;                    
                    sindex[step] = rcount[step];
                    step++;
                    wsize /= 2;
                }
                mpi_errno = MPIC_Recv(recvbuf, rcount[nsteps - 1], datatype, 0,
                                      MPIR_REDUCE_TAG, comm_ptr,
                                      MPI_STATUS_IGNORE, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }                
                newrank = 0;                
            } else if (newrank == 0) {
                /* Send a data to the root */
                mpi_errno = MPIC_Send(recvbuf, rcount[nsteps - 1], datatype,
                                      root, MPIR_REDUCE_TAG, comm_ptr, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }                
                newrank = -1;
            }
        } else {
            /* Case 2: root < 2r and a root is even: newroot = root / 2 */
            newroot = root / 2;
        }
    } else {
        /* Case 3: root >= 2r: newroot = root - r */
        newroot = root - rem;
    }

    /*
     * Step 3. Gather result at the newroot by the binomial tree algorithm.
     * Each process has 1 / p' of the total reduction result:
     *   rcount[nsteps - 1] elements in the recvbuf[rindex[nsteps - 1]...].
     * All exchanges are executed in reverse order relative
     * to recursive doubling (previous step).
     */
    if (newrank != -1) {
        mask = pof2 >> 1;
        step = nsteps - 1; /* step = ilog2(p') - 1 */

        while (mask > 0) {
            newdst = newrank ^ mask;
            /* Find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 : newdst + rem;
            /* If root is playing the role of newdst=0, adjust for it */
            if ((newdst == 0) && (root < 2 * rem) && (root % 2 != 0))
                dst = root;

            /* If the root of newdst's half of the tree is the
               same as the root of newroot's half of the tree, 
               send to newdst and exit, else receive from newdst. */
            newdst_tree_root = newdst >> step;
            newdst_tree_root <<= step;
            newroot_tree_root = newroot >> step;
            newroot_tree_root <<= step;
            
            if (newdst_tree_root == newroot_tree_root) {
                /* Send data from recvbuf and exit */ 
                mpi_errno = MPIC_Send((char *)recvbuf + rindex[step] * extent,
                                      rcount[step], datatype, dst,
                                      MPIR_REDUCE_TAG, comm_ptr, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
                break;
            } else {
                /* Recv and continue */
                mpi_errno = MPIC_Recv((char *)recvbuf + sindex[step] * extent,
                                      scount[step], datatype, dst,
                                      MPIR_REDUCE_TAG, comm_ptr,
                                      MPI_STATUS_IGNORE, errflag);
                if (mpi_errno) {
                    /* for communication errors, just record the error but continue */
                    *errflag = MPIR_ERR_GET_CLASS(mpi_errno);
                    MPIR_ERR_SET(mpi_errno, *errflag, "**fail");
                    MPIR_ERR_ADD(mpi_errno_ret, mpi_errno);
                }
            }
            step--;
            mask >>= 1;
        }
    }

    /* FIXME does this need to be checked after each uop invocation for
       predefined operators? */
    /* --BEGIN ERROR HANDLING-- */
    if (MPID_THREADPRIV_FIELD(op_errno)) {
        mpi_errno = MPID_THREADPRIV_FIELD(op_errno);
        goto fn_fail;
    }
    /* --END ERROR HANDLING-- */

fn_exit:
    MPIU_CHKLMEM_FREEALL();
    if (mpi_errno_ret)
        mpi_errno = mpi_errno_ret;
    else if (*errflag != MPIR_ERR_NONE)
        MPIR_ERR_SET(mpi_errno, *errflag, "**coll_fail");
    return mpi_errno;
fn_fail:
    goto fn_exit;
}
