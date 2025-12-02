import argparse
import tarfile
import io
import soundfile as sf


def longest_wav_in_tar(tar_path):
    max_len = 0.0
    max_name = None

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.name.lower().endswith(".wav"):
                continue

            f = tf.extractfile(member)
            if f is None:
                continue

            data = io.BytesIO(f.read())

            with sf.SoundFile(data) as s:
                duration = len(s) / s.samplerate

            if duration > max_len:
                max_len = duration
                max_name = member.name

    return max_name, max_len


def main():
    parser = argparse.ArgumentParser(
        description="Find longest WAV file inside a .tar/.tar.gz archive."
    )
    parser.add_argument("archive", help="Path to tar or tar.gz file")

    args = parser.parse_args()

    fname, length = longest_wav_in_tar(args.archive)

    if fname is None:
        print("No WAV files found.")
    else:
        print(f"Longest file: {fname}")
        print(f"Duration: {length:.3f} seconds")


if __name__ == "__main__":
    main()
