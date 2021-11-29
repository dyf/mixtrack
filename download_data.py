import json
import pandas as pd
import requests
import os
import zipfile

def download_file(url, prefix, chunk_size=8192):
    local_filename = os.path.join(prefix, url.split('/')[-1])
    if os.path.exists(local_filename):
        return local_filename

    # NOTE the stream=True parameter below
    with requests.get(url, stream=False) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            f.write(r.content)
    return local_filename

FEATURES = [ 
    "Accordion",
    "Ambience",
    "Bass",
    "Brass",
    "Guitar, Acoustic",
    "Guitar, Electric",
    "Guitar, Nonspecific",
    "Guitar, Pedal/Lap",
    "Harmonica",
    "Harp",
    "Keyboards, Celeste",
    "Keyboards, Clavinet",
    "Keyboards, Mellotron",
    "Keyboards, Nonspecific",
    "Keyboards, Organ",
    "Keyboards, Piano",
    "Keyboards, Rhodes",
    "Keyboards, Synthesizer",
    "Keyboards, Wurlitzer",
    "Loops",
    "MIDI",
    "Microphone Specific",
    "Mixes",
    "Percussion, Drum Kit",
    "Percussion, Glasses",
    "Percussion, Mallet",
    "Percussion, Miscellaneous",
    "Percussion, Nonspecific",
    "Percussion, Orchestral",
    "Percussion, Shakers",
    "Percussion, World Drum",
    "Sound Effects",
    "Strings, Folk",
    "Strings, Orchestral",
    "Vocals",
    "Woodwinds"
]

FEATURE_IDX = { v:i for i,v in enumerate(FEATURES) }

def download(input_json, output_dir, output_csv):
    with open(input_json,'r') as f:
        data = json.load(f)
            
    df = pd.DataFrame.from_records(data)

    df.rename(columns = {'p':'project', 'a':'artist', 'pt':'project_type', 'pv': 'preview', 
                        't': 'tracks', 'na':'new_string', 'dls':'download_size', 'd':'difficulty', 
                        'dl':'download_string', 'f':'fid', 'tc': 'thread_count', 'e': 'explicit', 'os': 'on_site',
                        'ic': 'features'}, inplace = True)

    print(f"all projects: {len(df)}")
    df = df[df.project_type.isin(["Full"])]#,"Excerpt"])] # excerpts are dups
    print(f"full projects: {len(df)}")
    #df = df[df.features.apply(lambda x: x[34]>0)]
    #print(len(df))
    df = df[df.on_site == "On"]
    print(f"on site projects: {len(df)}")
    url = df.download_string.apply(lambda x: x.split('"')[1])
    url.name = "url"
    df = df.merge(url, left_index=True, right_index=True)

    file_names = []
    for i,r in df.iterrows():    
        file_name = download_file(r.url, output_dir)
        print(i,file_name)
        file_name = os.path.abspath(file_name)
        file_names.append(file_name)

    file_names = pd.Series(file_names, index=df.index, name="file_name")

    df = df.merge(file_names, left_index=True, right_index=True)
    df.to_csv(output_csv)

    return df

def list_tracks(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zf:
        file_names = zf.namelist()
        wav_files = [ f for f in file_names if f.endswith(".wav") ]
        return wav_files

if __name__ == "__main__": 
    df = download(input_json="projects.json", 
                  output_dir="D:\\multitrack music", 
                  output_csv="metadata.csv")

    tracks = list_tracks(df.iloc[18].file_name)

    print(tracks)
