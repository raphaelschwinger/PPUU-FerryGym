# Anaconda Dev Container

Can be used as a [VS Code devcontainer](https://code.visualstudio.com/docs/remote/containers).

## How to access GUI

Open browser at http://localhost:8080/vnc.html.
You can also use VS Code "Simple Browser" (search in Command Palette).

## How to add dependency

### OS (Debian) depenency

Install from container shell

```bash
sudo apt-get update && sudo apt-get install [package-name(s)]
```

If it works, add it to `Dockerfile`.

### Conda dependencies

Install from container shell

```bash
conda install [package-name(s)]
```

If it works, export explicit list of packages.

```bash
conda list --explicit > .devcontainer/explicit-list.txt
```

Note, `conda` can be quite slow. 
You can use [mamba](https://github.com/mamba-org/mamba) as well, it is a lot faster.
