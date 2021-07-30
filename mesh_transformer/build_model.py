import functools
import multiprocessing

import optax
import ray

from mesh_transformer import util
from mesh_transformer.TPU_cluster import TPUCluster
from mesh_transformer.transformer_shard import CausalTransformer, CausalTransformerV2
from mesh_transformer.util import clip_by_global_norm, additive_weight_decay
# from ray_tpu import create_tpu, wait_til, get_connection, start_ray
from ray_tpu import TPU_RESOURCE


def build_model(params, version=1):
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    cores_per_replica = params["cores_per_replica"]
    tpu_size = params["tpu_size"]

    warmup_steps = params["warmup_steps"]
    anneal_steps = params["anneal_steps"]
    lr = params["lr"]
    end_lr = params["end_lr"]
    weight_decay = params["weight_decay"]

    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        additive_weight_decay(weight_decay),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(warmup_steps, anneal_steps, lr, end_lr))
    )

    params["optimizer"] = opt

    if version == 2:
        model_fn = functools.partial(CausalTransformerV2, params)
    elif version == 1:
        model_fn = functools.partial(CausalTransformer, params)
    else:
        raise Exception(f"Version {version} does not exist")

    t = TPUCluster((tpu_size // cores_per_replica, cores_per_replica), ray.cluster_resources()[TPU_RESOURCE], model_fn, version=version)
    return t
