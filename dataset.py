import musdb
import random
import tensorflow as tf

class MusdbData:
    def __init__(self, musdb_root):
        self.db = musdb.DB(root=musdb_root)    
        self.test_db = musdb.DB(root=musdb_root, subsets=["test"])    
        self.train_db = musdb.DB(root=musdb_root, subsets=["train"])    

    def random_iterator(self, N, chunk_duration, subset=None):
        if subset == "test":
            db = self.test_db
        elif subset == "train":
            db = self.train_db
        else:
            db = self.db

        p2 = 1
        
        for i in range(N):
            track = random.choice(db.tracks)
            track.chunk_duration = chunk_duration
            track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
            
            mix_track = track.audio
            vocal_track = track.targets['vocals'].audio
            accompaniment_track = track.targets['accompaniment'].audio

            if p2 == 1:
                while p2 < mix_track.shape[0]:
                    p2 *= 2
                p2 /= 2
                p2 = int(p2)

            yield (
                mix_track[:p2,:], 
                (
                    vocal_track[:p2,:], 
                    accompaniment_track[:p2,:]
                )
            )

    def random_dataset(self, N, chunk_duration, subset=None):
        def gen():
            yield from self.random_iterator(N, chunk_duration, subset)

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None,2), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(None,2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,2), dtype=tf.float32),
                )
            )
        )

if __name__ == "__main__":
    ds = MusdbData('D:/MUSDB18/Full')
    tfds = ds.random_dataset(4, 5.0)
    print(tfds)
    print(list(tfds.take(1)))
