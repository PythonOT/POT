Development installation
------------------------

If you want to develop cudamat, you should clone the github repository instead
of downloading a release. Furthermore, it is useful to install it in editable
mode. Instead of copying the files somewhere Python can find them, this will
point Python directly to the directory you install it from. Either of the
following commands will do:

```bash
# a) Install for your user in editable mode:
python setup.py develop --prefix=~/.local
# b) Install for your user in editable mode, but with pip:
pip install --user --editable .
```

As for the [standard installation](INSTALL.md), you can set the `NVCC_FLAGS`
environment variable to compile for a specific architecture.

Update after local changes
--------------------------

Your changes to `.py` files will show up immediately the next time you import
cudamat. Changes to `.cu` and `.cuh` files require a recompilation triggered
by just running the above installation command again.

Update after remote changes
---------------------------

To obtain the latest version, just pull in the remote changes:

```bash
git checkout master
git fetch origin
git merge origin/master
```

Then recompile as per the instructions in the previous section.

Contribute back
---------------

If you created a great new feature that is useful to the rest of the world,
and it even comes with docstrings and updated tests, we will gladly incorporate
it into cudamat. To do that, you will need to send us a pull request from your
fork.

If you haven't forked cudamat yet, log in to your github account, go to
https://github.com/cudamat/cudamat and hit the "Fork" button.
Now instead of having `origin` point to `cudamat/cudamat`, you will want to have
it point to your fork, and have `upstream` point to the official project:

```bash
git remote rename origin upstream
git remote add origin git@github.com:yourusername/cudamat
git fetch origin
```

Create a branch to house your changes:

```bash
git checkout -b my-new-feature
```

Hack away, then add your changes, commit and push:

```bash
git add all.py my.py changes.py
git commit -m 'Added a new feature that does this and that'
git push origin my-new-feature
```

Now send us a pull request asking to merge `yourusername:my-new-feature` into
`cudamat:master` and we will come back to you!
