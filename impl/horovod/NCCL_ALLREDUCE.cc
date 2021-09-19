void NCCLAllreduce::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto& ev : event_set) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
    }
  }
}

Status NCCLAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  auto& first_entry = entries[0];

  gpu_op_context_.InitGPU(entries);
  nccl_op_context_.InitNCCLComm(entries, response.devices());
  gpu_op_context_.InitGPUQueue(entries, response);

  WaitForData(entries);

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy (and possibly scale) tensors into the fusion buffer.
  if (entries.size() > 1) {
    ScaleMemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len, response.prescale_factor());
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, MEMCPY_IN_FUSION_BUFFER, *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
    int64_t num_elements = buffer_len / DataType_Size(first_entry.tensor->dtype());
    if (response.prescale_factor() != 1.0) {
      // Execute prescaling op
      ScaleBuffer(response.prescale_factor(), entries, fused_input_data, buffer_data, num_elements);
      fused_input_data = buffer_data; // for unfused, scale is done out of place
    }
  }

  // Do allreduce.
  int64_t num_elements = buffer_len / DataType_Size(first_entry.tensor->dtype());
  auto nccl_result = ncclAllReduce(fused_input_data, buffer_data,
                                   (size_t) num_elements,
                                   GetNCCLDataType(first_entry.tensor), ncclSum,
                                   *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
  nccl_context_->ErrorCheck("ncclAllReduce", nccl_result, *nccl_op_context_.nccl_comm_);
  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLREDUCE, *gpu_op_context_.stream);
  }

  // Copy (and possible scale) tensors out of the fusion buffer.
  if (entries.size() > 1) {
    ScaleMemcpyOutFusionBuffer(buffer_data, buffer_len, response.postscale_factor(), entries);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, MEMCPY_OUT_FUSION_BUFFER, *gpu_op_context_.stream);
    }
  } else {
    if (response.postscale_factor() != 1.0) {
      // Execute postscaling op
      ScaleBuffer(response.postscale_factor(), entries, buffer_data, buffer_data, num_elements);
    }
  }

  return gpu_op_context_.FinalizeGPUQueue(entries, true, nccl_op_context_.error_check_callback_);
}