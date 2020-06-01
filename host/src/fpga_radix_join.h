#ifndef FPGA_PARALLEL_RADIX_JOIN_H
#define FPGA_PARALLEL_RADIX_JOIN_H
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "types.h" /* relation_t */

#include "prj_params.h"

extern cl_platform_id platform;
extern cl_device_id device;
extern cl_context context;
extern cl_command_queue queue_relRead;
extern cl_command_queue queue_gather;
extern cl_command_queue queue_sgather;
extern cl_command_queue queue_filter[NUM_FPGA_DATAPATH];
extern cl_command_queue queue_sfilter[NUM_FPGA_DATAPATH];

extern cl_command_queue queue_hashjoin;
extern cl_kernel kernel_hashjoin;
extern cl_kernel kernel_gather;
extern cl_kernel kernel_filter[NUM_FPGA_DATAPATH];
extern cl_kernel kernel_sgather;
extern cl_kernel kernel_sfilter[NUM_FPGA_DATAPATH];

extern cl_kernel kernel_relRead;
extern cl_program program;
extern cl_int status;

extern unsigned int * rTable;
extern unsigned int * sTable;
extern unsigned int * HashTable;
extern unsigned int * matchedTable;
extern unsigned int * rHashCount;

extern int factor;
extern unsigned int rTupleNum; //= 1024*256;//0x1000000/factor;//16318*1024; //16 * 1024 * 1204 ;
extern unsigned int sTupleNum;// = 1024*256;//16318*1024; //16 * 1024 * 1024;
extern unsigned int rHashTableBucketNum;// = 4 * 1024 * 1024 / factor; //32*1024; //0x400000; //
extern unsigned int hashBucketSize;//      = 4*rTupleNum / rHashTableBucketNum;
extern unsigned int rTableSize;// = sizeof(unsigned int)*2*rTupleNum;
extern unsigned int sTableSize;// = sizeof(unsigned int)*2*sTupleNum;
extern unsigned int matchedTableSize;// = rTableSize + sTableSize;
extern unsigned int * rTableReadRange;
extern unsigned int * sTableReadRange;

int64_t
RJ_FPGA(relation_t * relR, relation_t * relS, int nthreads);
void 
dump_error(const char *str, cl_int status); 
int 
setHardwareEnv(cl_uint &num_platforms, cl_uint &num_devices);
void freeResources();
int creatKernels();
int varMap();
int varUnmap();

#endif
