import torch
import numpy as np
from numpy.linalg import norm


class Memory(object):
    """
        Create the empty memory buffer
    """

    def __init__(self, size, dimension=1 * 3 * 512 * 512):
        self.memory = {}
        self.size = size    # size=40
        self.dimension = dimension  # dimension=75

    def reset(self):
        self.memory = {}

    def get_size(self):
        return len(self.memory)

    def push(self, keys, logits):   # keys=[1,1024,320], logits=[1,3,512,512]
        for i, key in enumerate(keys):
            if len(self.memory.keys()) > self.size:
                self.memory.pop(list(self.memory)[0])
                # memory里面总共有40个
            self.memory.update(
                {key.reshape(self.dimension).tobytes(): (logits[i])})

    def _prepare_batch(self, sample, attention_weight):
        attention_weight = np.array(attention_weight / 0.2)
        attention_weight = np.exp(attention_weight) / (np.sum(np.exp(attention_weight)))# attention_weight=(16,)
        ensemble_prediction = sample[0] * attention_weight[0]   # sample[0]=(3,5,5)
        for i in range(1, len(sample)):
            ensemble_prediction = ensemble_prediction + sample[i] * attention_weight[i]

        return torch.FloatTensor(ensemble_prediction)   # ensemble_prediction=(3,5,5)

    def get_neighbours(self, keys, k):  # k=16, key=(1,3,5,5)
        """
        Returns samples from buffer using nearest neighbour approach
        """
        samples = []

        keys = keys.reshape(len(keys), self.dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, self.dimension)
        # self.all_keys = (41,75)
        for key in keys:    # keys=(1,75)
            similarity_scores = np.dot(self.all_keys, key.T) / (norm(self.all_keys, axis=1) * norm(key.T))
            # similarity_scores=(41,)
            K_neighbour_keys = self.all_keys[np.argpartition(similarity_scores, -k)[-k:]]   # K_neighbour_keys=(16,75)
            neighbours = [self.memory[nkey.tobytes()] for nkey in K_neighbour_keys]
            # neighbors=16*[3,5,5]
            attention_weight = np.dot(K_neighbour_keys, key.T) / (norm(K_neighbour_keys, axis=1) * norm(key.T))
            batch = self._prepare_batch(neighbours, attention_weight)   # attention_Weight=(16,) , batch=[3,5,5]
            samples.append(batch)

        return torch.stack(samples), np.mean(similarity_scores)# samples=(3,5,5), similarity_scores=(41,)
