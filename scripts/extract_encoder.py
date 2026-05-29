"""Extract vision encoder params from a checkpoint into a standalone pkl file.

Usage:
    python scripts/extract_encoder.py <checkpoint_path> <output_path>

Example:
    python scripts/extract_encoder.py \
        checkpoints/ogpo/.../params_500000.pkl \
        checkpoints/ogpo/square_encoder.pkl
"""

import pickle
import sys


def extract_encoder(checkpoint_path: str, output_path: str):
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)

    agent_dict = ckpt['agent']

    # Format 1: actor_network (from OGPO/BPTT full training)
    if 'actor_network' in agent_dict:
        params = agent_dict['actor_network']['params']
        if 'modules_actor' in params and 'encoder' in params['modules_actor']:
            encoder_params = params['modules_actor']['encoder']
        else:
            raise ValueError("No encoder found in modules_actor")

    # Format 2: network with modules_actor_bc_flow_encoder (from BC-only pretraining)
    elif 'network' in agent_dict:
        params = agent_dict['network']['params']
        if 'modules_actor_bc_flow_encoder' in params:
            encoder_params = params['modules_actor_bc_flow_encoder']
        else:
            raise ValueError("No encoder found in network params")
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(agent_dict.keys())}")

    with open(output_path, 'wb') as f:
        pickle.dump(encoder_params, f)

    # Print summary
    import jax
    num_params = sum(x.size for x in jax.tree.leaves(encoder_params))
    print(f"Extracted encoder: {num_params:,} parameters")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python scripts/extract_encoder.py <checkpoint_path> <output_path>")
        sys.exit(1)
    extract_encoder(sys.argv[1], sys.argv[2])
