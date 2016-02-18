#FCUDA: CUDA-to-FPGA Flow

##Introduction
- FCUDA is a source-to-source transformation framework that takes CUDA kernels with FCUDA annotation
pragmas as input and produces a synthesizable C code. The C code then can be ported to a High-level Synthesis
tool (e.g. Vivado HLS) to generate RTL for execution on FPGA.

- FCUDA is written largely based on Cetus - a source-to-source compiler infrastructure for C program
which is a research work from Purdue University. The ancestor of FCUDA is MCUDA. It has been largely 
modified and added some necessary compiler passes on top of it to facilitate the translation of
CUDA kernel to synthesizable C code. That said, FCUDA is also a source-to-source compiler (CUDA to C) and
does not rely on any specific compiler infrastructure from NVIDIA (nvcc).

##CUDA Support
- FCUDA can only support CUDA C 4.5 at this moment. C++ syntax and texture memory is not yet supported.

##FCUDA Setup
- Change the directory to *fcuda_src/*. The compiler is written in Java. Hence, Java compiler 
(javac) is required. Please install openjdk. Alternatively, you can run the script: `python setup_packages.py` 
to automatically install all the required software.

- Compile the source code: `./make_FCUDA.sh compile` . It will compile Cetus source code (located under
*cetus-1.3/src/*) and FCUDA source code (located under *src/*) to generate binary files **cetus.jar** and
**fcuda_jar**, respectively. These files are put under *jar/*. Please note that whenever you update the 
FCUDA source code or Cetus source code, you have to do the recompilation.

- Open the script **env_vars.sh** at the top-level directory. Add the path of FCUDA and other software
according to the setup of your system in this file. After that, execute the command `source env_vars.sh` to
export all the defined environment variables.

##FCUDA C-Level Testing
- Clone the FCUDA benchmark repository.

- The repository contains several benchmarks for testing FCUDA at C-Level. They are CUDA applications
collected from some CUDA benchmark suites such as NVIDIA SDK, Parboil suite, Rodinia suite.
These applications have been largely modified to removed or replaced all the CUDA specific syntax with 
"equivalent" C syntax, e.g. cudaMalloc --> malloc, cudaMemcpy --> memcpy, etc. Also, the CUDA kernel
invocation is changed to the (generated) C kernel invocation. The purpose is to verify whether the
generated C code (from FCUDA) is functionally correct, i.e. providing correct result as the original 
CUDA code.

- Change the directory to one particular benchmark, e.g. matmul.

- Run: `./fcuda_gen.sh` to translate the CUDA kernel code to C code. The CUDA kernel code
is located under *kernel/*, whereas the new generated C code will be put under *fcuda_gen/*.

- Run: `make` to compile the generated C code with other files.

- Run: `run.sh` to execute the benchmark.

- Each benchmark will have verification code at the end: either verifying the result produced by the
generated C kernel with given golden output or gold CPU implementation. If the output is "PASSED",
the generated C code is correct. If the output is "FAILED", it means FCUDA does not generate a functionally
correct C code: either there is a bug in the FCUDA compiler or FCUDA pragmas are not added properly.

##FCUDA Pragma HOWTO:

- There are some rules for inserting FCUDA pragma annotations on a CUDA kernel input file.

- First thing we have to do is adding the pragma GRID before a CUDA kernel function:

```c
#pragma FCUDA GRID x_dim=16 y_dim=16
__global__ void cuda_kernel( ... )
{
	stmt;
	...
}
```

- Basically, it tells FCUDA that this CUDA kernel uses 2-dimensional CUDA Grid, each dimension
has the thread block dimension of size 16. FCUDA uses this information to iniatialize a BLOCKDIM
constants which might be used in the generated code.

- Now it's time to add the main FCUDA pragma annotations: COMPUTE & TRANSFER. FCUDA will translate
the CUDA kernel into a C code with COMPUTE and TRANSFER tasks (functions) as we indicated when adding
the corresponding pragma annotations.

- A TRANSFER task is to copy data based on many assignment statements of the CUDA kernel input. Before
we talk much into this task, it's worth mentioning a bit about mapping CUDA memory space to FPGA memory
space. CUDA memory space consists of several kinds of memory, but we only care about 2 types in this
context: Global memory and Shared memory. Global memory is visible for all threads of all thread blocks,
but the latency is slow, whereas Shared memory is seen by threads within a thread block and each thread
block has one Shared memory. The latency of Shared memory is faster than Global memory's. To utilize this
fact, we often copy data from Global memory into Shared memory before doing any computation on the data,
and then copy it back when everything is done. Likewise, FPGA memory space can also divide into 2 kinds
of memory: off-chip memory (refers to DRAM), and on-chip memory (refers to BRAM). Accessing to on-chip
memory is also faster than to off-chip memory. Thus, since the two devices' memory spaces are quite
analogous, we can map CUDA's global memory to FPGA's off-chip memory, and CUDA's shared memory to FPGA's
on-chip memory. Therefore, to identify statements of a CUDA kernel which should be FCUDA TRANSFER task,
we should look for any assignment statement which assigns a shared-memory array element to a global-memory
array element or vice-versa, and wrap the FCUDA TRANSFER annotations around it to guide FCUDA to convert
it to a TRANSFER task. The assignment statement will be convert to `memcpy` function to utilize the memory
bursting feature of FPGA which is quite efficient in copying bulk of data at a time.

- Everything else which does not belong to FCUDA TRANSFER task can theoretically convert to COMPUTE task.
We can have a single COMPUTE task or multiple COMPUTE tasks based on how many regions of code we wrap them
with FCUDA COMPUTE pragma annotations. One must investigate the logic of the CUDA kernel code in order to
splitting the tasks makes sense and works correctly. Please note that for now, a variable which is used
in different COMPUTE tasks will be privatized, or convert to array. This is a technique to make sure that
each thread has a correct version of its private variable, since we serialize everything here, and this is
in the context of C, not CUDA. This could be changed in the future as it is not suffice to rely on this
reason to privatizing the variable, in some cases we also need to privatize it even in the same COMPUTE task.

- The remaining code which is neither inside COMPUTE nor TRANSFER tasks will be put under the main function
of FCUDA generated file. This main function basically serializes executing thread blocks in ForLoop fashion.
Each loop iteration executes one thread block, and inside the iteration it will invoke the COMPUTE and
TRANSFER tasks we have specified. These tasks will serially execute threads in ForLoop fashion. Each loop
iteration executes one thread. Therefore, the remaining code should not have any threadIdx-related
statements. They should all be moved under COMPUTE or TRANSFER task, otherwise it will make no sense.

- To achieve better performance, sometimes we need to hand-edit the CUDA kernel. Basically we need to avoid
doing computation on off-chip memory (refers to CUDA Global memory) at any cost, or we should never have
access to off-chip memory within a COMPUTE task. We should replace it with an on-chip memory. If it is
not available in the code, we have to introduce it ourselves. Off-chip memory should be accessed from within
TRANSFER tasks during fetching (off-chip memory to on-chip memory) or writing (on-chip memory to off-chip
memory), but there is also an exception when there is a condition on an assignment statement between
on-chip vs. off-chip. That condition may only allow assignment to occur on some specific elements and
their positions in memory are not in contiguous block at all. We better wrap this whole piece of code with
COMPUTE pragma in this situation to make sure the generated code functions correct.

- As an example, this is a sample code with some FCUDA pragma annotations:

```c
#pragma FCUDA TRANSFER cores=[1] type=burst begin name=fetch dir=[0] pointer=[off_chip_ptr] size=[BLOCKDIM] unroll=1 mpart=1 array_split=[on_chip_ptr]
	on_chip_ptr[threadIdx.x] = off_chip_ptr[threadIdx.x + offset_off_chip];
#pragma FCUDA TRANSFER cores=[1] type=burst end name=fetch dir=[0] pointer=[off_chip_ptr] size=[BLOCKDIM] unroll=1 mpart=1 array_split=[on_chip_ptr]

#pragma FCUDA COMPUTE cores=[1] begin name=compute unroll=1 mpart=1 array_split=[on_chip_ptr]
	stmt1;
	stmt2;
	...
#pragma FCUDA COMPUTE cores=[1] end name=compute unroll=1 mpart=1 array_split=[on_chip_ptr]

#pragma FCUDA TRANSFER cores=[1] type=burst begin name=write dir=[1] pointer=[off_chip_ptr] size=[BLOCKDIM] unroll=1 mpart=1 array_split=[on_chip_ptr]
	off_chip_ptr[threadIdx.x + offset_off_chip] = on_chip_ptr[threadIdx.x];
#pragma FCUDA TRANSFER cores=[1] type=burst begin name=write dir=[1] pointer=[off_chip_ptr] size=[BLOCKDIM] unroll=1 mpart=1 array_split=[on_chip_ptr]

```

- A FCUDA pragma annotation begins with a pragma begin, and ends with a pragma end. The two pragma annotations
are identical except for one has "begin" and the other has "end" which marks the region of code belong to that
FCUDA pragma type. Later FCUDA will translate this region of code based on the information supplied by the
FCUDA pragma annotations.

- A FCUDA TRANSFER pragma has the following configuration:
	+ cores: the number of duplicated task.
	+ type: type of TRANSFER. There is two types: **stream** for handling transferring constant data, and
		**burst** for normal assignment statements. Please see benchmark **cp** if you want to learn
		about **stream**.
	+ begin/end: mark the beginning/ending of a FCUDA task.
	+ name: name of the task, it will be added a prefix which is the name of the CUDA kernel function to
		name the task (function) after FCUDA finishes translation.
	+ dir: direction of TRANSFER. "0" indicates a transfer from off-chip memory to off-chip memory, and
	       "1" indicates a transfer from on-chip memory to off-chip memory.
	+ pointer: an off-chip memory pointer involving in this TRANSFER task.
	+ size: the amount of data assigned to `memcpy`.
	+ unroll: the degree of unrolling thread loop.
	+ mpart: the degree of partitioning an array.
	+ array_split: the array to be partitioned.

- A FCUDA COMPUTE pragma has the following configuration:
	+ cores: same as TRANSFER.
	+ begin/end: same as TRANSFER.
	+ name: same as TRANSFER.
	+ unroll, mpart, array_split: same as TRANSFER.

- An ideal model of a FCUDA generated code should have the skeleton of 3 tasks: TRANSFER (fetch), COMPUTE
(compute), TRANSFER (write). Of course, not all CUDA kernels can be translated into FCUDA C code with
this skeleton. Some of them will be much more complex. And this model also does not guarantee the most
optimization of the FCUDA generated code. One must study the CUDA kernel thoroughly to make decision
of how to insert FCUDA pragma annotations in a smart way to have a good performance.

- For more information and better insight on FCUDA, you can read an old README file in `fcuda_src/README.kg`.
This file lists several issues of FCUDA, some has been fixed and some has not to dated.

##Relevant Publication
+ [1] MCUDA: An Efficient Implementation of CUDA Kernels for Multi-Core CPUs (John et al.).
+ [2] FCUDA: Enabling Efficient Compilation of CUDA Kernels onto FPGAs (Alex et al.).
+ [3] Multilevel Granularity Parallelism Synthesis on FPGAs (Alex et al.).
+ [4] Throughput-oriented Kernel Porting onto FPGAs (Alex et al.).
+ [5] Integrated CUDA-to-FPGA Synthesis with Network-on-Chip (Swathi et al.).
+ [6] FCUDA-NoC: A Scalable and Efficient Network-on-Chip Implementation for the CUDA-to-FPGA Flow (Yao Chen et al.).

##Contact
- Please contact Tan at tan.nguyen@adsc.com.sg if you have any question, feedback, or bug report.
