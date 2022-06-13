import torch


class NCLMemoryModule:
    """
    A simple object to store the *M* most recent training instances from the previous batches.
    In short, this is a FIFO queue.
    This is used during the transformation, where this queue is used to have a larger pool of data from which we
    can pick the closest instance and create more realistic transformations.
    """
    def __init__(self, device, M=2000, labeled_memory=False):
        self.labeled_memory = labeled_memory
        self.current_update_idx = 0
        self.device = device
        self.queue_size = M

        self.data_memory = torch.Tensor([]).to(device)  # => The memory of the encoded data
        self.original_data_memory = torch.Tensor([]).to(device)  # => The memory of the original data
        if labeled_memory is True:
            self.labels_memory = torch.tensor([], dtype=torch.int32, device=device)

    def memory_step(self, input_data, input_original_data, input_labels=None):
        batch_size = input_data.shape[0]
        # If the memory queue isn't full yet, concatenate the batch to complete it
        if len(self.data_memory) < self.queue_size:
            len_to_cat = min(self.queue_size - len(self.data_memory), batch_size)

            self.data_memory = torch.cat((self.data_memory, input_data[:len_to_cat]))
            self.original_data_memory = torch.cat((self.original_data_memory, input_original_data[:len_to_cat]))
            if input_labels is not None:
                self.labels_memory = torch.cat((self.labels_memory, input_labels[:len_to_cat]))

            # If we have leftovers after concatenation, update memory
            if len_to_cat < batch_size:
                self.update_queue(batch_size - len_to_cat, input_data[len_to_cat:], input_original_data[len_to_cat:],
                                  input_labels[len_to_cat:] if input_labels is not None else None)
        else:
            # If the memory was full to begin with, update memory
            self.update_queue(batch_size, input_data, input_original_data, input_labels)

    def update_queue(self, batch_size, new_data, new_original_data, new_labels=None):
        indexes_to_update = torch.arange(batch_size).to(self.device)
        indexes_to_update += self.current_update_idx
        indexes_to_update = torch.fmod(indexes_to_update, self.queue_size)  # Previously incorrect : torch.fmod(indexes_to_update, *batch_size*) !!!

        self.data_memory.index_copy_(0, indexes_to_update, new_data)
        self.original_data_memory.index_copy_(0, indexes_to_update, new_original_data)
        if new_labels is not None:
            self.labels_memory.index_copy_(0, indexes_to_update, new_labels)

        self.current_update_idx = (self.current_update_idx + batch_size) % self.queue_size
