"""Tests for gromov._dictionary.py"""

# Author: Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#
# License: MIT License

import numpy as np

import ot


def test_gromov_wasserstein_linear_unmixing(nx):
    n = 4

    X1, y1 = ot.datasets.make_data_classif("3gauss", n, random_state=42)
    X2, y2 = ot.datasets.make_data_classif("3gauss2", n, random_state=42)

    C1 = ot.dist(X1)
    C2 = ot.dist(X2)
    Cdict = np.stack([C1, C2])
    p = ot.unif(n)

    C1b, C2b, Cdictb, pb = nx.from_numpy(C1, C2, Cdict, p)

    tol = 10 ** (-5)
    # Tests without regularization
    reg = 0.0
    unmixing1, C1_emb, OT, reconstruction1 = (
        ot.gromov.gromov_wasserstein_linear_unmixing(
            C1,
            Cdict,
            reg=reg,
            p=p,
            q=p,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=20,
            max_iter_inner=200,
        )
    )

    unmixing1b, C1b_emb, OTb, reconstruction1b = (
        ot.gromov.gromov_wasserstein_linear_unmixing(
            C1b,
            Cdictb,
            reg=reg,
            p=None,
            q=None,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=20,
            max_iter_inner=200,
        )
    )

    unmixing2, C2_emb, OT, reconstruction2 = (
        ot.gromov.gromov_wasserstein_linear_unmixing(
            C2,
            Cdict,
            reg=reg,
            p=None,
            q=None,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=20,
            max_iter_inner=200,
        )
    )

    unmixing2b, C2b_emb, OTb, reconstruction2b = (
        ot.gromov.gromov_wasserstein_linear_unmixing(
            C2b,
            Cdictb,
            reg=reg,
            p=pb,
            q=pb,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=20,
            max_iter_inner=200,
        )
    )

    np.testing.assert_allclose(unmixing1, nx.to_numpy(unmixing1b), atol=5e-06)
    np.testing.assert_allclose(unmixing1, [1.0, 0.0], atol=5e-01)
    np.testing.assert_allclose(unmixing2, nx.to_numpy(unmixing2b), atol=5e-06)
    np.testing.assert_allclose(unmixing2, [0.0, 1.0], atol=5e-01)
    np.testing.assert_allclose(C1_emb, nx.to_numpy(C1b_emb), atol=1e-06)
    np.testing.assert_allclose(C2_emb, nx.to_numpy(C2b_emb), atol=1e-06)
    np.testing.assert_allclose(
        reconstruction1, nx.to_numpy(reconstruction1b), atol=1e-06
    )
    np.testing.assert_allclose(
        reconstruction2, nx.to_numpy(reconstruction2b), atol=1e-06
    )
    np.testing.assert_allclose(C1b_emb.shape, (n, n))
    np.testing.assert_allclose(C2b_emb.shape, (n, n))

    # Tests with regularization

    reg = 0.001
    unmixing1, C1_emb, OT, reconstruction1 = (
        ot.gromov.gromov_wasserstein_linear_unmixing(
            C1,
            Cdict,
            reg=reg,
            p=p,
            q=p,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=20,
            max_iter_inner=200,
        )
    )

    unmixing1b, C1b_emb, OTb, reconstruction1b = (
        ot.gromov.gromov_wasserstein_linear_unmixing(
            C1b,
            Cdictb,
            reg=reg,
            p=None,
            q=None,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=20,
            max_iter_inner=200,
        )
    )

    unmixing2, C2_emb, OT, reconstruction2 = (
        ot.gromov.gromov_wasserstein_linear_unmixing(
            C2,
            Cdict,
            reg=reg,
            p=None,
            q=None,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=20,
            max_iter_inner=200,
        )
    )

    unmixing2b, C2b_emb, OTb, reconstruction2b = (
        ot.gromov.gromov_wasserstein_linear_unmixing(
            C2b,
            Cdictb,
            reg=reg,
            p=pb,
            q=pb,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=20,
            max_iter_inner=200,
        )
    )

    np.testing.assert_allclose(unmixing1, nx.to_numpy(unmixing1b), atol=1e-06)
    np.testing.assert_allclose(unmixing1, [1.0, 0.0], atol=1e-01)
    np.testing.assert_allclose(unmixing2, nx.to_numpy(unmixing2b), atol=1e-06)
    np.testing.assert_allclose(unmixing2, [0.0, 1.0], atol=1e-01)
    np.testing.assert_allclose(C1_emb, nx.to_numpy(C1b_emb), atol=1e-06)
    np.testing.assert_allclose(C2_emb, nx.to_numpy(C2b_emb), atol=1e-06)
    np.testing.assert_allclose(
        reconstruction1, nx.to_numpy(reconstruction1b), atol=1e-06
    )
    np.testing.assert_allclose(
        reconstruction2, nx.to_numpy(reconstruction2b), atol=1e-06
    )
    np.testing.assert_allclose(C1b_emb.shape, (n, n))
    np.testing.assert_allclose(C2b_emb.shape, (n, n))


def test_gromov_wasserstein_dictionary_learning(nx):
    # create dataset composed from 2 structures which are repeated 5 times
    shape = 4
    n_samples = 2
    n_atoms = 2
    projection = "nonnegative_symmetric"
    X1, y1 = ot.datasets.make_data_classif("3gauss", shape, random_state=42)
    X2, y2 = ot.datasets.make_data_classif("3gauss2", shape, random_state=42)
    C1 = ot.dist(X1)
    C2 = ot.dist(X2)
    Cs = [C1.copy() for _ in range(n_samples // 2)] + [
        C2.copy() for _ in range(n_samples // 2)
    ]
    ps = [ot.unif(shape) for _ in range(n_samples)]
    q = ot.unif(shape)

    # Provide initialization for the graph dictionary of shape (n_atoms, shape, shape)
    # following the same procedure than implemented in gromov_wasserstein_dictionary_learning.
    dataset_means = [C.mean() for C in Cs]
    rng = np.random.RandomState(0)
    Cdict_init = rng.normal(
        loc=np.mean(dataset_means),
        scale=np.std(dataset_means),
        size=(n_atoms, shape, shape),
    )

    if projection == "nonnegative_symmetric":
        Cdict_init = 0.5 * (Cdict_init + Cdict_init.transpose((0, 2, 1)))
        Cdict_init[Cdict_init < 0.0] = 0.0

    Csb = nx.from_numpy(*Cs)
    psb = nx.from_numpy(*ps)
    qb, Cdict_initb = nx.from_numpy(q, Cdict_init)

    # Test: compare reconstruction error using initial dictionary and dictionary learned using this initialization
    # > Compute initial reconstruction of samples on this random dictionary without backend
    use_adam_optimizer = True
    verbose = False
    tol = 10 ** (-5)
    epochs = 1

    initial_total_reconstruction = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Cs[i],
            Cdict_init,
            p=ps[i],
            q=q,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        initial_total_reconstruction += reconstruction

    # > Learn the dictionary using this init
    Cdict, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Cs,
        D=n_atoms,
        nt=shape,
        ps=ps,
        q=q,
        Cdict_init=Cdict_init,
        epochs=epochs,
        batch_size=2 * n_samples,
        learning_rate=1.0,
        reg=0.0,
        tol_outer=tol,
        tol_inner=tol,
        max_iter_outer=10,
        max_iter_inner=50,
        projection=projection,
        use_log=False,
        use_adam_optimizer=use_adam_optimizer,
        verbose=verbose,
    )
    # > Compute reconstruction of samples on learned dictionary without backend
    total_reconstruction = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Cs[i],
            Cdict,
            p=None,
            q=None,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction += reconstruction

    np.testing.assert_array_less(total_reconstruction, initial_total_reconstruction)

    # Test: Perform same experiments after going through backend

    Cdictb, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Csb,
        D=n_atoms,
        nt=shape,
        ps=None,
        q=None,
        Cdict_init=Cdict_initb,
        epochs=epochs,
        batch_size=n_samples,
        learning_rate=1.0,
        reg=0.0,
        tol_outer=tol,
        tol_inner=tol,
        max_iter_outer=10,
        max_iter_inner=50,
        projection=projection,
        use_log=False,
        use_adam_optimizer=use_adam_optimizer,
        verbose=verbose,
    )
    # Compute reconstruction of samples on learned dictionary
    total_reconstruction_b = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Csb[i],
            Cdictb,
            p=psb[i],
            q=qb,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction_b += reconstruction

    total_reconstruction_b = nx.to_numpy(total_reconstruction_b)
    np.testing.assert_array_less(total_reconstruction_b, initial_total_reconstruction)
    np.testing.assert_allclose(total_reconstruction_b, total_reconstruction, atol=1e-05)
    np.testing.assert_allclose(total_reconstruction_b, total_reconstruction, atol=1e-05)
    np.testing.assert_allclose(Cdict, nx.to_numpy(Cdictb), atol=1e-03)

    # Test: Perform same comparison without providing the initial dictionary being an optional input
    #       knowing than the initialization scheme is the same than implemented to set the benchmarked initialization.
    Cdict_bis, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Cs,
        D=n_atoms,
        nt=shape,
        ps=None,
        q=None,
        Cdict_init=None,
        epochs=epochs,
        batch_size=n_samples,
        learning_rate=1.0,
        reg=0.0,
        tol_outer=tol,
        tol_inner=tol,
        max_iter_outer=10,
        max_iter_inner=50,
        projection=projection,
        use_log=False,
        use_adam_optimizer=use_adam_optimizer,
        verbose=verbose,
        random_state=0,
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_bis = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Cs[i],
            Cdict_bis,
            p=ps[i],
            q=q,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction_bis += reconstruction

    np.testing.assert_allclose(
        total_reconstruction_bis, total_reconstruction, atol=1e-05
    )

    # Test: Same after going through backend
    Cdictb_bis, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Csb,
        D=n_atoms,
        nt=shape,
        ps=psb,
        q=qb,
        Cdict_init=None,
        epochs=epochs,
        batch_size=n_samples,
        learning_rate=1.0,
        reg=0.0,
        tol_outer=tol,
        tol_inner=tol,
        max_iter_outer=10,
        max_iter_inner=50,
        projection=projection,
        use_log=False,
        use_adam_optimizer=use_adam_optimizer,
        verbose=verbose,
        random_state=0,
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_b_bis = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Csb[i],
            Cdictb_bis,
            p=None,
            q=None,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction_b_bis += reconstruction

    total_reconstruction_b_bis = nx.to_numpy(total_reconstruction_b_bis)
    np.testing.assert_allclose(
        total_reconstruction_b_bis, total_reconstruction_b, atol=1e-05
    )
    np.testing.assert_allclose(Cdict_bis, nx.to_numpy(Cdictb_bis), atol=1e-03)

    # Test: Perform same comparison without providing the initial dictionary being an optional input
    #       and testing other optimization settings untested until now.
    #       We pass previously estimated dictionaries to speed up the process.
    use_adam_optimizer = False
    verbose = True
    use_log = True

    Cdict_bis2, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Cs,
        D=n_atoms,
        nt=shape,
        ps=ps,
        q=q,
        Cdict_init=Cdict,
        epochs=epochs,
        batch_size=n_samples,
        learning_rate=10.0,
        reg=0.0,
        tol_outer=tol,
        tol_inner=tol,
        max_iter_outer=10,
        max_iter_inner=50,
        projection=projection,
        use_log=use_log,
        use_adam_optimizer=use_adam_optimizer,
        verbose=verbose,
        random_state=0,
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_bis2 = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Cs[i],
            Cdict_bis2,
            p=ps[i],
            q=q,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction_bis2 += reconstruction

    np.testing.assert_array_less(total_reconstruction_bis2, total_reconstruction)

    # Test: Same after going through backend
    Cdictb_bis2, log = ot.gromov.gromov_wasserstein_dictionary_learning(
        Csb,
        D=n_atoms,
        nt=shape,
        ps=psb,
        q=qb,
        Cdict_init=Cdictb,
        epochs=epochs,
        batch_size=n_samples,
        learning_rate=10.0,
        reg=0.0,
        tol_outer=tol,
        tol_inner=tol,
        max_iter_outer=10,
        max_iter_inner=50,
        projection=projection,
        use_log=use_log,
        use_adam_optimizer=use_adam_optimizer,
        verbose=verbose,
        random_state=0,
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_b_bis2 = 0
    for i in range(n_samples):
        _, _, _, reconstruction = ot.gromov.gromov_wasserstein_linear_unmixing(
            Csb[i],
            Cdictb_bis2,
            p=psb[i],
            q=qb,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction_b_bis2 += reconstruction

    total_reconstruction_b_bis2 = nx.to_numpy(total_reconstruction_b_bis2)
    np.testing.assert_allclose(
        total_reconstruction_b_bis2, total_reconstruction_bis2, atol=1e-05
    )


def test_fused_gromov_wasserstein_linear_unmixing(nx):
    n = 4
    X1, y1 = ot.datasets.make_data_classif("3gauss", n, random_state=42)
    X2, y2 = ot.datasets.make_data_classif("3gauss2", n, random_state=42)
    F, y = ot.datasets.make_data_classif("3gauss", n, random_state=42)

    C1 = ot.dist(X1)
    C2 = ot.dist(X2)
    Cdict = np.stack([C1, C2])
    Ydict = np.stack([F, F])
    p = ot.unif(n)

    C1b, C2b, Fb, Cdictb, Ydictb, pb = nx.from_numpy(C1, C2, F, Cdict, Ydict, p)

    # Tests without regularization
    reg = 0.0

    unmixing1, C1_emb, Y1_emb, OT, reconstruction1 = (
        ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            C1,
            F,
            Cdict,
            Ydict,
            p=p,
            q=p,
            alpha=0.5,
            reg=reg,
            tol_outer=10 ** (-6),
            tol_inner=10 ** (-6),
            max_iter_outer=10,
            max_iter_inner=50,
        )
    )

    unmixing1b, C1b_emb, Y1b_emb, OTb, reconstruction1b = (
        ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            C1b,
            Fb,
            Cdictb,
            Ydictb,
            p=None,
            q=None,
            alpha=0.5,
            reg=reg,
            tol_outer=10 ** (-6),
            tol_inner=10 ** (-6),
            max_iter_outer=10,
            max_iter_inner=50,
        )
    )

    unmixing2, C2_emb, Y2_emb, OT, reconstruction2 = (
        ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            C2,
            F,
            Cdict,
            Ydict,
            p=None,
            q=None,
            alpha=0.5,
            reg=reg,
            tol_outer=10 ** (-6),
            tol_inner=10 ** (-6),
            max_iter_outer=10,
            max_iter_inner=50,
        )
    )

    unmixing2b, C2b_emb, Y2b_emb, OTb, reconstruction2b = (
        ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            C2b,
            Fb,
            Cdictb,
            Ydictb,
            p=pb,
            q=pb,
            alpha=0.5,
            reg=reg,
            tol_outer=10 ** (-6),
            tol_inner=10 ** (-6),
            max_iter_outer=10,
            max_iter_inner=50,
        )
    )

    np.testing.assert_allclose(unmixing1, nx.to_numpy(unmixing1b), atol=4e-06)
    np.testing.assert_allclose(unmixing1, [1.0, 0.0], atol=4e-01)
    np.testing.assert_allclose(unmixing2, nx.to_numpy(unmixing2b), atol=4e-06)
    np.testing.assert_allclose(unmixing2, [0.0, 1.0], atol=4e-01)
    np.testing.assert_allclose(C1_emb, nx.to_numpy(C1b_emb), atol=1e-03)
    np.testing.assert_allclose(C2_emb, nx.to_numpy(C2b_emb), atol=1e-03)
    np.testing.assert_allclose(Y1_emb, nx.to_numpy(Y1b_emb), atol=1e-03)
    np.testing.assert_allclose(Y2_emb, nx.to_numpy(Y2b_emb), atol=1e-03)
    np.testing.assert_allclose(
        reconstruction1, nx.to_numpy(reconstruction1b), atol=1e-06
    )
    np.testing.assert_allclose(
        reconstruction2, nx.to_numpy(reconstruction2b), atol=1e-06
    )
    np.testing.assert_allclose(C1b_emb.shape, (n, n))
    np.testing.assert_allclose(C2b_emb.shape, (n, n))

    # Tests with regularization
    reg = 0.001

    unmixing1, C1_emb, Y1_emb, OT, reconstruction1 = (
        ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            C1,
            F,
            Cdict,
            Ydict,
            p=p,
            q=p,
            alpha=0.5,
            reg=reg,
            tol_outer=10 ** (-6),
            tol_inner=10 ** (-6),
            max_iter_outer=10,
            max_iter_inner=50,
        )
    )

    unmixing1b, C1b_emb, Y1b_emb, OTb, reconstruction1b = (
        ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            C1b,
            Fb,
            Cdictb,
            Ydictb,
            p=None,
            q=None,
            alpha=0.5,
            reg=reg,
            tol_outer=10 ** (-6),
            tol_inner=10 ** (-6),
            max_iter_outer=10,
            max_iter_inner=50,
        )
    )

    unmixing2, C2_emb, Y2_emb, OT, reconstruction2 = (
        ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            C2,
            F,
            Cdict,
            Ydict,
            p=None,
            q=None,
            alpha=0.5,
            reg=reg,
            tol_outer=10 ** (-6),
            tol_inner=10 ** (-6),
            max_iter_outer=10,
            max_iter_inner=50,
        )
    )

    unmixing2b, C2b_emb, Y2b_emb, OTb, reconstruction2b = (
        ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            C2b,
            Fb,
            Cdictb,
            Ydictb,
            p=pb,
            q=pb,
            alpha=0.5,
            reg=reg,
            tol_outer=10 ** (-6),
            tol_inner=10 ** (-6),
            max_iter_outer=10,
            max_iter_inner=50,
        )
    )

    np.testing.assert_allclose(unmixing1, nx.to_numpy(unmixing1b), atol=1e-06)
    np.testing.assert_allclose(unmixing1, [1.0, 0.0], atol=1e-01)
    np.testing.assert_allclose(unmixing2, nx.to_numpy(unmixing2b), atol=1e-06)
    np.testing.assert_allclose(unmixing2, [0.0, 1.0], atol=1e-01)
    np.testing.assert_allclose(C1_emb, nx.to_numpy(C1b_emb), atol=1e-03)
    np.testing.assert_allclose(C2_emb, nx.to_numpy(C2b_emb), atol=1e-03)
    np.testing.assert_allclose(Y1_emb, nx.to_numpy(Y1b_emb), atol=1e-03)
    np.testing.assert_allclose(Y2_emb, nx.to_numpy(Y2b_emb), atol=1e-03)
    np.testing.assert_allclose(
        reconstruction1, nx.to_numpy(reconstruction1b), atol=1e-06
    )
    np.testing.assert_allclose(
        reconstruction2, nx.to_numpy(reconstruction2b), atol=1e-06
    )
    np.testing.assert_allclose(C1b_emb.shape, (n, n))
    np.testing.assert_allclose(C2b_emb.shape, (n, n))


def test_fused_gromov_wasserstein_dictionary_learning(nx):
    # create dataset composed from 2 structures which are repeated 5 times
    shape = 4
    n_samples = 2
    n_atoms = 2
    projection = "nonnegative_symmetric"
    X1, y1 = ot.datasets.make_data_classif("3gauss", shape, random_state=42)
    X2, y2 = ot.datasets.make_data_classif("3gauss2", shape, random_state=42)
    F, y = ot.datasets.make_data_classif("3gauss", shape, random_state=42)

    C1 = ot.dist(X1)
    C2 = ot.dist(X2)
    Cs = [C1.copy() for _ in range(n_samples // 2)] + [
        C2.copy() for _ in range(n_samples // 2)
    ]
    Ys = [F.copy() for _ in range(n_samples)]
    ps = [ot.unif(shape) for _ in range(n_samples)]
    q = ot.unif(shape)

    # Provide initialization for the graph dictionary of shape (n_atoms, shape, shape)
    # following the same procedure than implemented in gromov_wasserstein_dictionary_learning.
    dataset_structure_means = [C.mean() for C in Cs]
    rng = np.random.RandomState(0)
    Cdict_init = rng.normal(
        loc=np.mean(dataset_structure_means),
        scale=np.std(dataset_structure_means),
        size=(n_atoms, shape, shape),
    )
    if projection == "nonnegative_symmetric":
        Cdict_init = 0.5 * (Cdict_init + Cdict_init.transpose((0, 2, 1)))
        Cdict_init[Cdict_init < 0.0] = 0.0
    dataset_feature_means = np.stack([Y.mean(axis=0) for Y in Ys])
    Ydict_init = rng.normal(
        loc=dataset_feature_means.mean(axis=0),
        scale=dataset_feature_means.std(axis=0),
        size=(n_atoms, shape, 2),
    )

    Csb = nx.from_numpy(*Cs)
    Ysb = nx.from_numpy(*Ys)
    psb = nx.from_numpy(*ps)
    qb, Cdict_initb, Ydict_initb = nx.from_numpy(q, Cdict_init, Ydict_init)

    # Test: Compute initial reconstruction of samples on this random dictionary
    alpha = 0.5
    use_adam_optimizer = True
    verbose = False
    tol = 1e-05
    epochs = 1

    initial_total_reconstruction = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Cs[i],
            Ys[i],
            Cdict_init,
            Ydict_init,
            p=ps[i],
            q=q,
            alpha=alpha,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        initial_total_reconstruction += reconstruction

    # > Learn a dictionary using this given initialization and check that the reconstruction loss
    # on the learned dictionary is lower than the one using its initialization.
    Cdict, Ydict, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Cs,
        Ys,
        D=n_atoms,
        nt=shape,
        ps=ps,
        q=q,
        Cdict_init=Cdict_init,
        Ydict_init=Ydict_init,
        epochs=epochs,
        batch_size=n_samples,
        learning_rate_C=1.0,
        learning_rate_Y=1.0,
        alpha=alpha,
        reg=0.0,
        tol_outer=tol,
        tol_inner=tol,
        max_iter_outer=10,
        max_iter_inner=50,
        projection=projection,
        use_log=False,
        use_adam_optimizer=use_adam_optimizer,
        verbose=verbose,
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Cs[i],
            Ys[i],
            Cdict,
            Ydict,
            p=None,
            q=None,
            alpha=alpha,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction += reconstruction
    # Compare both
    np.testing.assert_array_less(total_reconstruction, initial_total_reconstruction)

    # Test: Perform same experiments after going through backend
    Cdictb, Ydictb, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Csb,
        Ysb,
        D=n_atoms,
        nt=shape,
        ps=None,
        q=None,
        Cdict_init=Cdict_initb,
        Ydict_init=Ydict_initb,
        epochs=epochs,
        batch_size=2 * n_samples,
        learning_rate_C=1.0,
        learning_rate_Y=1.0,
        alpha=alpha,
        reg=0.0,
        tol_outer=tol,
        tol_inner=tol,
        max_iter_outer=10,
        max_iter_inner=50,
        projection=projection,
        use_log=False,
        use_adam_optimizer=use_adam_optimizer,
        verbose=verbose,
        random_state=0,
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_b = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Csb[i],
            Ysb[i],
            Cdictb,
            Ydictb,
            p=psb[i],
            q=qb,
            alpha=alpha,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction_b += reconstruction

    total_reconstruction_b = nx.to_numpy(total_reconstruction_b)
    np.testing.assert_array_less(total_reconstruction_b, initial_total_reconstruction)
    np.testing.assert_allclose(total_reconstruction_b, total_reconstruction, atol=1e-05)
    np.testing.assert_allclose(Cdict, nx.to_numpy(Cdictb), atol=1e-03)
    np.testing.assert_allclose(Ydict, nx.to_numpy(Ydictb), atol=1e-03)

    # Test: Perform similar experiment without providing the initial dictionary being an optional input
    Cdict_bis, Ydict_bis, log = ot.gromov.fused_gromov_wasserstein_dictionary_learning(
        Cs,
        Ys,
        D=n_atoms,
        nt=shape,
        ps=None,
        q=None,
        Cdict_init=None,
        Ydict_init=None,
        epochs=epochs,
        batch_size=n_samples,
        learning_rate_C=1.0,
        learning_rate_Y=1.0,
        alpha=alpha,
        reg=0.0,
        tol_outer=tol,
        tol_inner=tol,
        max_iter_outer=10,
        max_iter_inner=50,
        projection=projection,
        use_log=False,
        use_adam_optimizer=use_adam_optimizer,
        verbose=verbose,
        random_state=0,
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_bis = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Cs[i],
            Ys[i],
            Cdict_bis,
            Ydict_bis,
            p=ps[i],
            q=q,
            alpha=alpha,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction_bis += reconstruction

    np.testing.assert_allclose(
        total_reconstruction_bis, total_reconstruction, atol=1e-05
    )

    # > Same after going through backend
    Cdictb_bis, Ydictb_bis, log = (
        ot.gromov.fused_gromov_wasserstein_dictionary_learning(
            Csb,
            Ysb,
            D=n_atoms,
            nt=shape,
            ps=None,
            q=None,
            Cdict_init=None,
            Ydict_init=None,
            epochs=epochs,
            batch_size=n_samples,
            learning_rate_C=1.0,
            learning_rate_Y=1.0,
            alpha=alpha,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
            projection=projection,
            use_log=False,
            use_adam_optimizer=use_adam_optimizer,
            verbose=verbose,
            random_state=0,
        )
    )

    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_b_bis = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Csb[i],
            Ysb[i],
            Cdictb_bis,
            Ydictb_bis,
            p=psb[i],
            q=qb,
            alpha=alpha,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction_b_bis += reconstruction

    total_reconstruction_b_bis = nx.to_numpy(total_reconstruction_b_bis)
    np.testing.assert_allclose(
        total_reconstruction_b_bis, total_reconstruction_b, atol=1e-05
    )

    # Test: without using adam optimizer, with log and verbose set to True
    use_adam_optimizer = False
    verbose = True
    use_log = True

    # > Experiment providing previously estimated dictionary to speed up the test compared to providing initial random init.
    Cdict_bis2, Ydict_bis2, log = (
        ot.gromov.fused_gromov_wasserstein_dictionary_learning(
            Cs,
            Ys,
            D=n_atoms,
            nt=shape,
            ps=ps,
            q=q,
            Cdict_init=Cdict,
            Ydict_init=Ydict,
            epochs=epochs,
            batch_size=n_samples,
            learning_rate_C=10.0,
            learning_rate_Y=10.0,
            alpha=alpha,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
            projection=projection,
            use_log=use_log,
            use_adam_optimizer=use_adam_optimizer,
            verbose=verbose,
            random_state=0,
        )
    )
    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_bis2 = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Cs[i],
            Ys[i],
            Cdict_bis2,
            Ydict_bis2,
            p=ps[i],
            q=q,
            alpha=alpha,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction_bis2 += reconstruction

    np.testing.assert_array_less(total_reconstruction_bis2, total_reconstruction)

    # > Same after going through backend
    Cdictb_bis2, Ydictb_bis2, log = (
        ot.gromov.fused_gromov_wasserstein_dictionary_learning(
            Csb,
            Ysb,
            D=n_atoms,
            nt=shape,
            ps=None,
            q=None,
            Cdict_init=Cdictb,
            Ydict_init=Ydictb,
            epochs=epochs,
            batch_size=n_samples,
            learning_rate_C=10.0,
            learning_rate_Y=10.0,
            alpha=alpha,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
            projection=projection,
            use_log=use_log,
            use_adam_optimizer=use_adam_optimizer,
            verbose=verbose,
            random_state=0,
        )
    )

    # > Compute reconstruction of samples on learned dictionary
    total_reconstruction_b_bis2 = 0
    for i in range(n_samples):
        _, _, _, _, reconstruction = ot.gromov.fused_gromov_wasserstein_linear_unmixing(
            Csb[i],
            Ysb[i],
            Cdictb_bis2,
            Ydictb_bis2,
            p=None,
            q=None,
            alpha=alpha,
            reg=0.0,
            tol_outer=tol,
            tol_inner=tol,
            max_iter_outer=10,
            max_iter_inner=50,
        )
        total_reconstruction_b_bis2 += reconstruction

    # > Compare results with/without backend
    total_reconstruction_b_bis2 = nx.to_numpy(total_reconstruction_b_bis2)
    np.testing.assert_allclose(
        total_reconstruction_bis2, total_reconstruction_b_bis2, atol=1e-05
    )
