# Lecture 27 — Program Profiling and Profile Guided Optimization (POGO)

## Roadmap

Two exercises: one for systemtap and one for profiler guided optimization.

## Systemtap (10 minutes)

I couldn't figure out how to run the real dtrace on my Debian machines (or on
the ECE machines), but I can run systemtap, which seems to have similar
functionality. You can read about SystemTap in this 2021 article: [Using the
SystemTap Dyninst runtime
environment](https://developers.redhat.com/blog/2021/04/16/using-the-systemtap-dyninst-runtime-environment#),
and there is a somewhat stale [beginners
guide](https://sourceware.org/systemtap/SystemTap_Beginners_Guide/), though I
had to use Google to figure out how to get the example to run today, using the
[release notes for 4.1](https://lwn.net/Articles/787810/).

Here's some Hello, World content:

```
sudo apt install linux-image-5.16.0-5-amd64-dbg linux-headers-5.16.0-5-amd64
sudo apt install systemtap

sudo stap -v -e 'probe oneshot { println("hello world") }'
sudo stap -ve 'probe kernel.function("icmp_reply") { println("icmp reply") }'
```

and here I invoke a more sophisticated script, which I fixed from the Beginners'
Guide, and committed to the course repo under `lectures/flipped/L27/iotime.stp`.

```
sudo stap iotime.stp -c 'grep foobar /var/cache/apt/archives/[put some file here]'
```

It should be possible to count the number of calls to `clone()`, the system call
to create threads in Linux, on the Lab 4 code, which is otherwise kind of hard
to do. Got to run it in the A4 directory though.

```
sudo stap ~/courses/ece459/lectures/flipped/L27/clone.stp -c '/home/plam/hacking/ece459-a4/target/release/lab4'
```

(The starter code for A4 is at
[https://git.uwaterloo.ca/ece459-rust-assignments/ece459-a4]).

How do you know what syscalls to probe? If you run your program under `strace`,
it'll tell you what syscalls you're invoking (and their arguments).

## Profiler Guided Optimization (35 minutes)

Students should be able to install rust with rustup and apply the "A Complete
Cargo Workflow" steps at
[https://doc.rust-lang.org/rustc/profile-guided-optimization.html] to the A4
starter code. I observed a noticeable speedup.

I should rerun this with hyperfine, but I was getting times of 16.8s, 21.3s, and
19.2s with the starter code and arguments `target/release/lab4 100 100 50000`.
Then I ran the profile generation code with the same args and it took 2m30; I
ran it once more with `100 100 5000` to collect some more data, quickly. Then,
recompiling using the profile data gave 14.7s, 14.5s, and 15.7s. The workflow
was pretty much exactly the same as in the Complete Cargo Workflow.

# After-action report, plam, 17 Mar 2023

Ran systemtap on my laptop. Needed to install latest kernel. The scripts pretty
much work, though clone doesn't work with lab4. That is, it says "no calls to
clone()". Does work on the L2 live demo code.

Did an strace demo.

Talked about PGO but did not actually do it (again, tried to get through things
quickly). Having students go through the steps should work, but again, it's
probably indeed 35 minutes.
