@def title = "Julia tricks"
This page is hidden, for a reason: it is just a list of tricks that I use but that I do not expect to be useful for the general public.

## Installing Julia kernel on jupyter

Jupyter notebooks are really useful. In order to be able to use them at their best, we must be able to add kernels with our own specifications.

### 1. Create a Julia Environment
The first step requires the creation of a Julia environment.
Let us assume that you are in a `test_environment` folder. You just have to create a new environment.
![create_env](/assets/julia-tricks/create_env.png)
Now you can add all the packages you want!
### 2. Add Jupyter Kernel
The second step requires to use `IJulia` to add your Julia kernel to Jupyter.
![add_kernel](/assets/julia-tricks/add_kernel.png)
More info can be found on the IJulia [Documentation](https://julialang.github.io/IJulia.jl/stable/manual/installation/#Installing-additional-Julia-kernels).

## Julia and VSCode
VSCode is a great IDE for Julia, the one I am currently using. What you need, in order to freely add virtual envs to the studd you are doing, is to add the correct path to your Julia installation, in the Julia extension settings
![set_vscode](/assets/julia-tricks/vscode_set_julia_path.png)


## Distributed computing

### Julia Envs & Distributed computing
If you want to use a Julia environment with distributed computing, it is not enough to start the `julia` REPL or run a script using the `--project` flag: if you add a process, this will use the mail `julia` environment.
Remember, when creating processes, that you should pay attention to the env you are using...with the flag `exeflags="--project=$(Base.active_project())"`