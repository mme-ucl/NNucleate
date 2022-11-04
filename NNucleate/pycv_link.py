import numpy as np


def write_cv_link(model, n_hid, n_layers, n_at, box_l, fname):
    """Function that writes an input file for the coupling with plumed, based on a model that is passed in the parameters.
    This function assumes the following architecture:
    - Embedding layer n_at x n_hid
    - N_layers GCLs
    - Edge layer n_hid x n_hid, ReLU, n_hid x n_hid
    - Node layer 2*n_hid x n_hid, ReLU, n_hid x n_hid
    - Node decoder n_hid x n_hid, ReLU, n_hid x n_hid
    - Graph decoder n_hid x n_hid, ReLU, n_hid x 1

    :param model: The model for which the input file shall be written. (only for graph-based models)
    :type model: GNNCV
    :param n_hid: Number of dimensions in the model latent space.
    :type n_hid: int
    :param n_layers: Number of GCL layers.
    :type n_layers: int
    :param n_at: Number of nodes in the graph.
    :type n_at: int
    :param box_l: Size of the simulation box of the system that is used in the MTD simulation.
    :type box_l: float
    :param fname: Name of file that is created.
    :type fname: str
    """

    with open(fname, "w") as f:
        # import statement
        f.writelines(
            [
                "import numpy as np\n",
                "import mdtraj as md\n",
                "import math\n",
                "import jax\n",
                "from functools import partial\n",
                "from jax import vmap\n",
                "from jax import grad, jit, jacobian\n",
                "from jax import lax, random, numpy as jnp\n",
                "from flax import linen as jnn\n",
            ]
        )

        # parameters
        ## embedding weights
        emb_weights = model.state_dict()["embedding.weight"]
        emb_bias = model.state_dict()["embedding.bias"]
        kernel = emb_weights.detach().cpu().numpy()
        kernel = np.transpose(kernel, (1, 0))
        f.write("\n\nembedding_weights = jnp.array([[\n\t")
        for i in range(len(kernel)):
            for j in range(len(kernel[0])):
                f.write("%f,  " % kernel[i][j])

            if i == len(kernel) - 1:
                f.write("]")
            else:
                f.write("],\n\t[")
        f.write("])\n")

        f.write("embedding_bias = jnp.array([\n\t")
        for i in range(len(emb_bias)):
            f.write("%f,  " % emb_bias[i])
        f.write("])\n")

        ## GCL layers

        for i in range(n_layers):
            ew_key = "gcl_%d.edge_mlp.0.weight" % i
            eb_key = "gcl_%d.edge_mlp.0.bias" % i
            ew_key2 = "gcl_%d.edge_mlp.2.weight" % i
            eb_key2 = "gcl_%d.edge_mlp.2.bias" % i

            edge_weights = model.state_dict()[ew_key]
            edge_bias = model.state_dict()[eb_key]
            edge_weights2 = model.state_dict()[ew_key2]
            edge_bias2 = model.state_dict()[eb_key2]

            kernel = edge_weights.detach().cpu().numpy()
            kernel = np.transpose(kernel, (1, 0))

            kernel2 = edge_weights2.detach().cpu().numpy()
            kernel2 = np.transpose(kernel2, (1, 0))
            f.write("\n\nGCL%d_edge_weights_1 = jnp.array([[\n\t" % i)
            for j in range(len(kernel)):
                for k in range(len(kernel[0])):
                    f.write("%f,  " % kernel[j][k])

                if j == len(kernel) - 1:
                    f.write("]")
                else:
                    f.write("],\n\t[")
            f.write("])\n")

            f.write("GCL%d_edge_bias_1 = jnp.array([\n\t" % i)
            for j in range(len(edge_bias)):
                f.write("%f,  " % edge_bias[j])
            f.write("])\n")

            f.write("\n\nGCL%d_edge_weights_2 = jnp.array([[\n\t" % i)
            for j in range(len(kernel2)):
                for k in range(len(kernel2[0])):
                    f.write("%f,  " % kernel2[j][k])

                if j == len(kernel2) - 1:
                    f.write("]")
                else:
                    f.write("],\n\t[")
            f.write("])\n")

            f.write("GCL%d_edge_bias_2 = jnp.array([\n\t" % i)
            for j in range(len(edge_bias2)):
                f.write("%f,  " % edge_bias2[j])
            f.write("])\n")

            # node weights
            ew_key = "gcl_%d.node_mlp.0.weight" % i
            eb_key = "gcl_%d.node_mlp.0.bias" % i
            ew_key2 = "gcl_%d.node_mlp.2.weight" % i
            eb_key2 = "gcl_%d.node_mlp.2.bias" % i

            node_weights = model.state_dict()[ew_key]
            node_bias = model.state_dict()[eb_key]
            node_weights2 = model.state_dict()[ew_key2]
            node_bias2 = model.state_dict()[eb_key2]

            kernel = node_weights.detach().cpu().numpy()
            kernel = np.transpose(kernel, (1, 0))

            kernel2 = node_weights2.detach().cpu().numpy()
            kernel2 = np.transpose(kernel2, (1, 0))
            f.write("\n\nGCL%d_node_weights_1 = jnp.array([[\n\t" % i)
            for j in range(len(kernel)):
                for k in range(len(kernel[0])):
                    f.write("%f,  " % kernel[j][k])

                if j == len(kernel) - 1:
                    f.write("]")
                else:
                    f.write("],\n\t[")
            f.write("])\n")

            f.write("GCL%d_node_bias_1 = jnp.array([\n\t" % i)
            for j in range(len(node_bias)):
                f.write("%f,  " % node_bias[j])
            f.write("])\n")

            f.write("\n\nGCL%d_node_weights_2 = jnp.array([[\n\t" % i)
            for j in range(len(kernel2)):
                for k in range(len(kernel2[0])):
                    f.write("%f,  " % kernel2[j][k])

                if j == len(kernel2) - 1:
                    f.write("]")
                else:
                    f.write("],\n\t[")
            f.write("])\n")

            f.write("GCL%d_node_bias_2 = jnp.array([\n\t" % i)
            for j in range(len(node_bias2)):
                f.write("%f,  " % node_bias2[j])
            f.write("])\n")

        ## node_dec
        node_dec_weights = model.state_dict()["node_dec.0.weight"]
        node_dec_bias = model.state_dict()["node_dec.0.bias"]
        node_dec_weights2 = model.state_dict()["node_dec.2.weight"]
        node_dec_bias2 = model.state_dict()["node_dec.2.bias"]

        kernel = node_dec_weights.detach().cpu().numpy()
        kernel = np.transpose(kernel, (1, 0))

        kernel2 = node_dec_weights2.detach().cpu().numpy()
        kernel2 = np.transpose(kernel2, (1, 0))

        f.write("\n\nnode_dec_weights_1 = jnp.array([[\n\t")
        for i in range(len(kernel)):
            for j in range(len(kernel[0])):
                f.write("%f,  " % kernel[i][j])
            if i == len(kernel) - 1:
                f.write("]")
            else:
                f.write("],\n\t[")
        f.write("])\n")
        f.write("node_dec_bias_1 = jnp.array([\n\t")
        for i in range(len(node_dec_bias)):
            f.write("%f,  " % node_dec_bias[i])
        f.write("])\n")
        f.write("\n\nnode_dec_weights_2 = jnp.array([[\n\t")
        for i in range(len(kernel2)):
            for j in range(len(kernel2[0])):
                f.write("%f,  " % kernel2[i][j])
            if i == len(kernel2) - 1:
                f.write("]")
            else:
                f.write("],\n\t[")
        f.write("])\n")
        f.write("node_dec_bias_2 = jnp.array([\n\t")
        for i in range(len(node_dec_bias2)):
            f.write("%f,  " % node_dec_bias2[i])
        f.write("])\n")

        ## graph_dec
        node_gr_weights = model.state_dict()["graph_dec.0.weight"]
        node_gr_bias = model.state_dict()["graph_dec.0.bias"]
        node_gr_weights2 = model.state_dict()["graph_dec.2.weight"]
        node_gr_bias2 = model.state_dict()["graph_dec.2.bias"]

        kernel = node_gr_weights.detach().cpu().numpy()
        kernel = np.transpose(kernel, (1, 0))

        kernel2 = node_gr_weights2.detach().cpu().numpy()
        kernel2 = np.transpose(kernel2, (1, 0))

        f.write("\n\nnode_gr_weights_1 = jnp.array([[\n\t")
        for i in range(len(kernel)):
            for j in range(len(kernel[0])):
                f.write("%f,  " % kernel[i][j])
            if i == len(kernel) - 1:
                f.write("]")
            else:
                f.write("],\n\t[")
        f.write("])\n")
        f.write("node_gr_bias_1 = jnp.array([\n\t")
        for i in range(len(node_gr_bias)):
            f.write("%f,  " % node_gr_bias[i])
        f.write("])\n")
        f.write("\n\nnode_gr_weights_2 = jnp.array([[\n\t")
        for i in range(len(kernel2)):
            for j in range(len(kernel2[0])):
                f.write("%f,  " % kernel2[i][j])
            if i == len(kernel2) - 1:
                f.write("]")
            else:
                f.write("],\n\t[")
        f.write("])\n")
        f.write("node_gr_bias_2 = jnp.array([\n\t")
        for i in range(len(node_gr_bias2)):
            f.write("%f,  " % node_gr_bias2[i])
        f.write("])\n")

        # helper functions
        # pbc
        f.writelines(
            [
                "\ndef pbc(trajectory, box_length):\n",
                "# function defining PBC\n",
                "    def transform(x):\n",
                "        if x >= box_length:\n",
                "            x -= box_length\n",
                "        if x <= 0:\n",
                "            x += box_length\n",
                "        return x\n",
                "\n",
                "    # Prepare the function for 2D mapping\n",
                "    func = np.vectorize(transform)\n",
                "\n",
                "    # map the function on all coordinates\n",
                "    trajectory.xyz[0] = func(trajectory.xyz[0])\n",
                "\n",
                "    return trajectory\n",
            ]
        )

        # scatter jax
        f.writelines(
            [
                "\n\ndef scatter_jax(input, dim, index, src):\n",
                "    # Works like PyTorch's scatter. See https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html\n",
                "\n",
                "    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))\n",
                "\n",
                "    _scatter = jax.lax.scatter_add\n",
                "    _scatter = partial(_scatter, dimension_numbers=dnums)\n",
                "    vmap_inner = partial(vmap, in_axes=(0, 0, 0), out_axes=0)\n",
                "    vmap_outer = partial(vmap, in_axes=(1, 1, 1), out_axes=1)\n",
                "\n",
                "    for idx in range(len(input.shape)):\n",
                "        if idx == dim:\n",
                "            pass\n",
                "        elif idx < dim:\n",
                "            _scatter = vmap_inner(_scatter)\n",
                "        else:\n",
                "            _scatter = vmap_outer(_scatter)\n",
                "\n",
                "    return _scatter(input, jnp.expand_dims(index, axis=-1), src)\n",
            ]
        )

        # jax unsorted segment sum
        f.writelines(
            [
                "\n\ndef j_unsorted_segment_sum(data, segment_ids, num_segments):\n",
                "    result_shape = (num_segments, len(data[0]))\n",
                "    result = jnp.zeros(result_shape)  # Init empty result tensor.\n",
                "    segment_ids = jnp.concatenate([segment_ids.reshape(len(segment_ids), -1) for i in range(len(data[0]))], axis=1)\n",
                "    result = scatter_jax(result.astype(float), 0, segment_ids, data)\n",
                "    return result\n",
            ]
        )

        # rc edges
        f.writelines(
            [
                "\n\ndef get_rc_edges(rc, traj):\n",
                "    n_at = len(traj.xyz[0])\n",
                "    row_list = []\n",
                "    col_list = []\n",
                "    frame_row = []\n",
                "    frame_col = []\n",
                "\n",
                "    dist = md.compute_neighborlist(traj, rc, frame=0)\n",
                "\n",
                "    for j in range(n_at):\n",
                "        for k in range(len(dist[j])):\n",
                "            frame_row.append(j)\n",
                "            frame_col.append(dist[j][k])\n",
                "\n",
                "    row_list.append(jnp.array(frame_row, dtype=int))\n",
                "    col_list.append(jnp.array(frame_col, dtype=int))\n",
                "\n",
                "    return row_list, col_list\n",
            ]
        )

        # jax model
        f.write("\n\n@jit\n")
        f.write("def jax_model(X, j_row, j_col):\n")

        f.writelines(
            [
                "    # Embedding/Encoding layer\n",
                "    variables = {'params': {'kernel': embedding_weights, 'bias': embedding_bias}}\n",
                "    j_fc = jnn.Dense(features=%d)\n" % n_hid,
                "    j_h = j_fc.apply(variables, X)\n",
                "\n",
                "\n",
                "\n",
                "    # repeat GCL n_layers times\n",
            ]
        )

        for i in range(n_layers):
            f.writelines(
                [
                    "    #layer %d\n" % i,
                    "    j_edge = j_h[j_row] - j_h[j_col]\n",
                    "    variables = {'params': {'kernel': GCL%d_edge_weights_1, 'bias': GCL%d_edge_bias_1}}\n"
                    % (i, i),
                    "    variables2 = {'params': {'kernel': GCL%d_edge_weights_2, 'bias': GCL%d_edge_bias_2}}\n"
                    % (i, i),
                    "\n",
                    "\n",
                    "    j_fc = jnn.Dense(features=%d)\n" % n_hid,
                    "    j_out = j_fc.apply(variables, j_edge)\n",
                    "    j_out = jnn.relu(j_out)\n",
                    "    j_out = j_fc.apply(variables2, j_out)\n",
                    "    j_e_out = j_out\n",
                    "    j_agg = j_unsorted_segment_sum(j_e_out, j_row, num_segments=len(j_h))\n",
                    "    j_agg = jnp.concatenate([j_h, j_agg], axis=1)\n",
                    "\n",
                    "    # [outC, inC] -> [inC, outC]\n",
                    "\n",
                    "\n",
                    "    variables = {'params': {'kernel': GCL%d_node_weights_1, 'bias': GCL%d_node_bias_1}}\n"
                    % (i, i),
                    "    variables2 = {'params': {'kernel': GCL%d_node_weights_2, 'bias': GCL%d_node_bias_2}}\n"
                    % (i, i),
                    "\n",
                    "    j_fc = jnn.Dense(features=%d)\n" % n_hid,
                    "    j_fc2 = jnn.Dense(features=%d)\n" % n_hid,
                    "\n",
                    "    j_out = j_fc.apply(variables, j_agg)\n",
                    "    j_out = jnn.relu(j_out)\n",
                    "    j_out = j_fc2.apply(variables2, j_out)\n",
                    "    j_h = j_out\n",
                ]
            )

        f.writelines(
            [
                "\n# Graph decoder\n",
                "    # [outC, inC] -> [inC, outC]\n",
                "\n",
                "    variables = {'params': {'kernel': node_dec_weights_1, 'bias': node_dec_bias_1}}\n",
                "    variables2 = {'params': {'kernel': node_dec_weights_2, 'bias': node_dec_bias_2}}\n",
                "\n",
                "    j_fc = jnn.Dense(features=%d)\n" % n_hid,
                "    j_fc2 = jnn.Dense(features=%d)\n" % n_hid,
                "\n",
                "    j_out = j_fc.apply(variables, j_h)\n",
                "    j_out = jnn.relu(j_out)\n",
                "    j_n_dec = j_fc2.apply(variables2, j_out)\n",
                "\n",
                "    j_n_dec = j_n_dec.reshape(-1, %d, %d)\n" % (n_at, n_hid),
                "    j_n_sum = jnp.sum(j_n_dec, axis=1)\n",
                "\n",
                "    # [outC, inC] -> [inC, outC]\n",
                "\n",
                "    variables = {'params': {'kernel': node_gr_weights_1, 'bias': node_gr_bias_1}}\n",
                "    variables2 = {'params': {'kernel': node_gr_weights_2, 'bias': node_gr_bias_2}}\n",
                "\n",
                "    j_fc = jnn.Dense(features=%d)\n" % n_hid,
                "    j_fc2 = jnn.Dense(features=1)\n",
                "\n",
                "    j_out = j_fc.apply(variables, j_n_sum)\n",
                "    j_out = jnn.relu(j_out)\n",
                "    j_pred = j_fc2.apply(variables2, j_out)\n",
            ]
        )

        f.write("    return j_pred[0]\n")

        # jax grad
        f.write("\ngrad_mod = jacobian(jax_model)\n")

        f.writelines(["\n",
                "t = md.Topology()\n",
                "t.add_chain()\n",
                't.add_residue("1", t.chain(0))\n',
                "for i in range(%d):\n" % n_at,
                '    t.add_atom("coll", md.element.hydrogen, t.residue(0))\n'])
        
        f.writelines(["vecs = np.zeros((1, 3, 3))\n",
                "vecs[0, 0] = np.array([%f, 0, 0])\n" % box_l,
                "vecs[0, 1] = np.array([0, %f, 0])\n" % box_l,
                "vecs[0, 2] = np.array([0, 0, %f])\n" % box_l,
                "\n",])
        # cv1 
        f.writelines(
            [
                "\n\ndef cv1(x):\n",
                "\n",
                "    x = x/10.0\n",
                "\n",
                "    fake_traj = md.Trajectory(x, t, unitcell_angles=[90.0, 90.0, 90.0], unitcell_lengths=[%f, %f, %f])\n"
                % (box_l, box_l, box_l),
                "\n",
                "    fake_traj.unitcell_vectors = vecs\n",
                "    fake_traj = pbc(fake_traj, %f)\n" % box_l,
                "    rows, cols = get_rc_edges(0.6, fake_traj[0])\n",
                "\n",
                "    if len(rows[0]) == 0:\n",
                "        return 0.0, np.zeros(x.shape)\n",
                "\n",
                "    X = jnp.array(x/%f)\n" % box_l,
                "    pred = jax_model(X, rows[0], cols[0])\n",
                "    gs = grad_mod(X, rows[0], cols[0])\n",
                "\n",
                "    return pred.item(), np.array(gs[0])*%f*10\n" % box_l,
            ]
        )
