@def title = "Julia on Android"

# Julia on Android: how to use Julia everywhere

in this blog post, I am going to show how to install a terminal emulator on your Android
device and run Julia on top of it. This blog post is based on the tutorial that can be found at the following links:

- [RMSRosas termux julia install instructions 2021](https://gist.github.com/caseykneale/fb5503d95b29c1e3bf167a192bf17420)
- [Running Julia on my android phone](https://www.linkedin.com/pulse/running-julia-my-android-phone-paresh-mathur)
- [Ubuntu in Termux](https://github.com/MFDGaming/ubuntu-in-termux)

## First step: install Termux

The chosen emulator is Termux, but we are not going to use the Play Store version, but rather the one given by [F-Droid](https://f-droid.org/en/).

After downloading F-Droid and installing it, you need to look for a Termux emulator usng the search bar.
After installing Termux in this way, you should be able to update your environment
```
apt-get update && apt-get upgrade -y
```
Install `wget`
```
apt-get install wget -y
```
Install `proot`
```
apt-get install proot -y
```
Install `git`
```
apt-get install git -y
```
Go to `home`  folder
```
cd ~
```
Download script to install Ubuntu
```
git clone https://github.com/MFDGaming/ubuntu-in-termux.git
```
Go to script folder
```
cd ubuntu-in-termux
```
Give permission to run script
```
chmod +x ubuntu.sh
```
Run the script
```
./ubuntu.sh -y
```
Now you can start Ubuntu
```
./startubuntu.sh
```

# Second step: install Julia

Download Julia binaries. Here is the download lonk at the last moment I updated the blog post
```
wget https://julialang-s3.julialang.org/bin/linux/aarch64/1.8/julia-1.8.2-linux-aarch64.tar.gz
```
Extract the downloaded archive
```
tar zxvf julia-1.8.1-linux-x86_64.tar.gz
```
Add Julia to your Path. Open your bashrc
```
nano ~/.bashrc
```
Add to the end the *Absolute* path to Julia, which will look like 
```
export PATH="$PATH:/root/julia-1.8.2
/bin"
```
Now source your bashrc
```
source ~/.bashrc
```
You can now start Julia from terminal, using simply
```
julia
```