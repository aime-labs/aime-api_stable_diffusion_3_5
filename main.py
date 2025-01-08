import argparse
import base64
import datetime
import io
import random
from pathlib import Path

import numpy as np
import torch
from aime_api_worker_interface import APIWorkerInterface
from PIL import Image
from sd3_impls import SD3LatentFormat
from sd3_infer import SD3Inferencer

WORKER_JOB_TYPE = "stable_diffusion_3.5"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d7"
VERSION = 0


class ProcessOutputCallback:
    def __init__(self, api_worker, inferencer, model_name):
        self.api_worker = api_worker
        self.inferencer = inferencer
        self.model_name = model_name
        self.job_data = None

    def process_output(
        self, latent_image, progress_step=100, finished=True, error=None, message=None
    ):
        if error:
            print("error")
            self.api_worker.send_progress(100, None, job_data=self.job_data)
            image = Image.fromarray(
                (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
            )
            self.api_worker.send_job_results(
                {"images": [image], "error": error, "model_name": self.model_name}
            )
            self.job_data = None
            return
        else:
            if not finished:
                step_factor = (
                    self.job_data.get("denoise") if self.job_data.get("image") else 1
                )
                total_steps = int(self.job_data.get("steps") * step_factor) + 3
                progress_info = round((progress_step) * 100 / total_steps)

                if self.api_worker.progress_data_received:
                    progress_data = {"progress_message": message}
                    if isinstance(
                        latent_image, torch.Tensor
                    ):  # Ensure it's a latent tensor
                        if self.job_data.get("provide_progress_images") == "decoded":
                            progress_data["progress_images"] = (
                                self.inferencer.vae_decode(
                                    SD3LatentFormat().process_out(latent_image)
                                )
                            )
                        elif self.job_data.get("provide_progress_images") == "latent":
                            progress_data["progress_images"] = (
                                SD3LatentFormat().decode_latent_to_preview(latent_image)
                            )

                    return self.api_worker.send_progress(
                        progress_info, progress_data, job_data=self.job_data
                    )
            else:
                if isinstance(latent_image, list):
                    image_list = latent_image
                elif isinstance(latent_image, torch.Tensor):
                    image_list = self.inferencer.vae_decode(
                        SD3LatentFormat().process_out(latent_image)
                    )
                else:
                    image_list = [latent_image]

                self.api_worker.send_progress(100, None, job_data=self.job_data or {})
                self.api_worker.send_job_results(
                    {"images": image_list, "model_name": self.model_name}
                )
                self.job_data = None


def load_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_server",
        type=str,
        default="http://0.0.0.0:7777",
        help="Address of the AIME API server",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False, help="ID of the GPU to be used"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data/stable-diffusion-3.5-large/sd3.5_large.safetensors",
        help="Model weights",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/data/stable-diffusion-3.5-large/",
        help="Destination of model weigths",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./output", help="Destination of output images"
    )
    parser.add_argument(
        "--api_auth_key",
        type=str,
        default=DEFAULT_WORKER_AUTH_KEY,
        required=False,
        help="API server worker auth key",
    )

    return parser.parse_args()


def convert_base64_string_to_image(base64_string, width, height):
    if base64_string:
        base64_data = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_data)

        with io.BytesIO(image_data) as buffer:
            image = Image.open(buffer)
            return image.resize((width, height), Image.LANCZOS)


def set_seed(job_data):
    seed = job_data.get("seed", -1)
    if seed == -1:
        random.seed(datetime.datetime.now().timestamp())
        seed = random.randint(1, 99999999)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    job_data["seed"] = seed
    return job_data


@torch.no_grad()
def main():
    args = load_flags()
    torch.cuda.set_device(args.gpu_id)
    api_worker = APIWorkerInterface(
        args.api_server,
        WORKER_JOB_TYPE,
        args.api_auth_key,
        args.gpu_id,
        world_size=1,
        rank=0,
        gpu_name=torch.cuda.get_device_name(),
        worker_version=VERSION,
    )
    inferencer = SD3Inferencer()
    inferencer.load(Path(args.model), None, 3.0, None, Path(args.model_dir))
    callback = ProcessOutputCallback(
        api_worker, inferencer, "stable-diffusion-3.5-large"
    )

    while True:
        try:
            job_data = api_worker.job_request()
            print(f'Processing job {job_data.get("job_id")}...', end="", flush=True)

            init_image = job_data.get("image")
            print(f"Job Data: {job_data}", end="", flush=True)
            batch_size = job_data.get("num_samples", 1)
            if init_image:
                if not isinstance(init_image, str):
                    raise ValueError(
                        f"Expected a base64 string for init_image, got {type(init_image)} with value {init_image}"
                    )
                init_image = convert_base64_string_to_image(
                    init_image, job_data.get("width"), job_data.get("height")
                )
            callback.job_data = job_data
            inferencer.gen_image(
                batch_size=batch_size,
                prompts=job_data.get("prompt"),
                width=job_data.get("width"),
                height=job_data.get("height"),
                steps=job_data.get("steps"),
                cfg_scale=job_data.get("cfg_scale"),
                seed=job_data.get("seed"),
                seed_type=job_data.get("seed_type", "rand"),
                init_image=init_image,
                out_dir="./output",
                denoise=job_data.get("denoise", 1.0),
                negative_prompt=job_data.get("negative_prompt"),
                callback=callback.process_output,
            )
            print("Done")

        except ValueError as exc:
            print("Error:", exc)
            callback.process_output(
                None, None, True, f"{exc}\nChange parameters and try again"
            )
            continue
        except torch.cuda.OutOfMemoryError as exc:
            print("Error:", exc)
            callback.process_output(
                None,
                None,
                True,
                f"{exc}\nReduce number of samples or image size and try again",
            )
            continue
        except OSError as exc:
            print("Error:", exc)
            callback.process_output(None, None, True, f"{exc}\nInvalid image file")
            continue


if __name__ == "__main__":
    main()
