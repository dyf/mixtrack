import musdb
import random
import tensorflow as tf

class MusdbData:
    def __init__(self, musdb_root, is_wav, target):        
        self.target = target
        
        self.db = musdb.DB(root=musdb_root, is_wav=is_wav)    
        self.train_db = musdb.DB(root=musdb_root, is_wav=is_wav, subsets="train", split="train")        
        self.val_db = musdb.DB(root=musdb_root, is_wav=is_wav, subsets="train", split="valid")
                    
    def random_iterator(self, N_tracks, samples_per_track, chunk_duration, subset=None, split=None, augment=True):
        if subset == "val":
            db = self.val_db
        elif subset == "train":
            db = self.train_db
        else:
            db = self.db

        p2 = 1
        
        for i in range(N_tracks):
            gain = random.uniform(0.25,1.25) if augment else 1.0
            ch_swap = random.random() > 0.5 if augment else False

            track = random.choice(db.tracks)
            track.chunk_duration = chunk_duration

            for j in range(samples_per_track):
                track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
                
                mix_track = track.audio
                target_track = track.targets[self.target].audio

                if p2 == 1:
                    while p2 < mix_track.shape[0]:
                        p2 *= 2
                    p2 /= 2
                    p2 = int(p2)
                
                if ch_swap:
                    yield (
                        mix_track[:p2,::-1] * gain,
                        target_track[:p2,::-1] * gain
                    )
                else:
                    yield (
                        mix_track[:p2,:] * gain,
                        target_track[:p2,:] * gain
                    )

    def random_dataset(self, N_tracks, samples_per_track, chunk_duration, subset=None, augment=True):
        def gen():
            yield from self.random_iterator(N_tracks, samples_per_track, chunk_duration, subset, augment)

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None,2), dtype=tf.float32),
                tf.TensorSpec(shape=(None,2), dtype=tf.float32),
            )
        )

if __name__ == "__main__":
    ds = MusdbData('D:/MUSDB18/Full', is_wav=False, target='drums')
    tfds = ds.random_ft_dataset(4, 5.0)    
    
    datum = list(tfds.take(1))[0][0].numpy()
    print(datum)

    print(datum.min(),datum.max())
