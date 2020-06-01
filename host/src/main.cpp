#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "prj_params.h"
#include "cpu_parallel_radix_join.h"  /* parallel radix joins: RJ, PRO, PRH, PRHO */
#include "fpga_radix_join.h" 

#define AOCL_ALIGNMENT 64
using namespace aocl_utils;
using namespace std;

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue_relRead;
cl_command_queue queue_hashjoin;
cl_command_queue queue_gather;
cl_command_queue queue_filter[NUM_FPGA_DATAPATH];
//cl_command_queue queue_sfilter[NUM_FPGA_DATAPATH];
//cl_command_queue queue_sgather;

cl_kernel kernel_filter[NUM_FPGA_DATAPATH];
//cl_kernel kernel_sfilter[NUM_FPGA_DATAPATH];

cl_kernel kernel_hashjoin;
cl_kernel kernel_gather;
//cl_kernel kernel_sgather;

cl_kernel kernel_relRead;
cl_program program;
cl_int status;

unsigned int * rTable = NULL;
unsigned int * sTable = NULL;
unsigned int * HashTable = NULL;
unsigned int * matchedTable = NULL;
unsigned int * rHashCount = NULL;
unsigned int * rTableReadRange = NULL;
unsigned int * sTableReadRange = NULL;

#define HARP 1
#define DE5 2
#define CPU 3

//#define PLATFORM HARP
 #define PLATFORM HARP
// #define PLATFORM CPU


uint partition_num = pow(2,NUM_RADIX_BITS);

uint rSizeInK = 512;
uint srRatio = 1;
//uint srRatio = 5;
//uint srRatio = 4;
//uint srRatio = 3;
//uint srRatio = 2;
//uint srRatio = 1;

#if PLATFORM == HARP
uint nthreads = 28;
#elif PLATFORM == DE5
uint nthreads = 28;
#endif

//uint partition_num_loopcount = 1;
//uint partition_num_loopcount = 8;
uint partition_num_loopcount = partition_num;

int shuffleEnable = 1;
//int shuffleEnable = 0;

uint one_r_partition_num = 1024*rSizeInK;
uint one_s_partition_num = 1024*rSizeInK*srRatio;

unsigned int rTupleNum = 128000000;//0x1000000;//partition_num * one_r_partition_num;//0x1000000/factor;//16318*1024; //16 * 1024 * 1204 ;
unsigned int sTupleNum = 128000000;//0x4000000 * 10;//partition_num * one_s_partition_num;

uint inp_num_radix_bits = NUM_RADIX_BITS;

// unsigned int rHashTableBucketNum = 4 * 1024 * 1024 / factor; //32*1024; //0x400000; //
// unsigned int hashBucketSize      = 4*rTupleNum / rHashTableBucketNum;

unsigned int rTableSize = sizeof(unsigned int)*2*rTupleNum;
unsigned int sTableSize = sizeof(unsigned int)*2*sTupleNum;
unsigned int matchedTableSize = 400;//rTableSize + sTableSize;

#define RAND_RANGE(N) ((float)rand() / ((float)RAND_MAX + 1) * (N))

int setKernelEnv(){
	queue_relRead = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateCommandQueue.", status);
		freeResources();
		return 1;
	}

	queue_hashjoin = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateCommandQueue.", status);
		freeResources();
		return 1;
	}

	queue_gather = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateCommandQueue.", status);
		freeResources();
		return 1;
	}
	for(int i = 0; i < NUM_FPGA_DATAPATH; i ++){
		queue_filter[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
		if(status != CL_SUCCESS) {
			dump_error("Failed clCreateCommandQueue.", status);
			freeResources();
			return 1;
		}
	}

	cl_int kernel_status;
	size_t binsize = 0;
	unsigned char * binary_file = loadBinaryFile("./shj.aocx", &binsize);
	if(!binary_file) {
		dump_error("Failed loadBinaryFile.", status);
		freeResources();
		return 1;
	}

	program = clCreateProgramWithBinary(
			context, 1, &device, &binsize, 
			(const unsigned char**)&binary_file, 
			&kernel_status, &status);
	if(status != CL_SUCCESS) {
		dump_error("Failed clCreateProgramWithBinary.", status);
		freeResources();
		return 1;
	}

	delete [] binary_file;

	// build the program
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	if(status != CL_SUCCESS) {
		dump_error("Failed clBuildProgram.", status);
		freeResources();
		return 1;
	}

	return 0;
}

int create_relation_pk_s(unsigned int *tuple_addr, int num_tuples)
{
	int i;
	for (i = 0; i < num_tuples; i++) {
		tuple_addr[2*i] = (i+1);   //+1
		tuple_addr[2*i+1] = (i+2);   //+2
	}
	// shuffle tuples of the relation using Knuth shuffle
	if(shuffleEnable){ 
		//for (i = num_tuples - 1; i > 0; i--) {   //knuth_shuflle
		for (i = num_tuples - 2; i > 0; i--) {   //knuth_shuflle. Exclude last 8 tuples from shuffling inputs, for endflag
			int  j  = RAND_RANGE(i);
			int tmp = tuple_addr[2*i];        //intkey_t tmp            = relation->tuples[i].key;
			tuple_addr[2*i] = tuple_addr[2*j];//relation->tuples[i].key = relation->tuples[j].key;
			tuple_addr[2*j] = tmp;            //relation->tuples[j].key = tmp;
		}
	}
	return 0;
}

int create_relation_pk(unsigned int *tuple_addr, int num_tuples)
{
	int i;
	for (i = 0; i < num_tuples; i++) {
		tuple_addr[2*i] = (i+1);   //+1
		tuple_addr[2*i+1] = (i+2);   //+2
	}
	// shuffle tuples of the relation using Knuth shuffle
	if(shuffleEnable){ 
		//for (i = num_tuples - 1; i > 0; i--) {   //knuth_shuflle
		// Exclude last 8 tuples from shuffling inputs, for endflag
		for (i = num_tuples - 1; i > 0; i--) {   //knuth_shuflle. 
			int  j  = RAND_RANGE(i);
			int tmp = tuple_addr[2*i];        //intkey_t tmp            = relation->tuples[i].key;
			tuple_addr[2*i] = tuple_addr[2*j];//relation->tuples[i].key = relation->tuples[j].key;
			tuple_addr[2*j] = tmp;            //relation->tuples[j].key = tmp;
		}
	}
	return 0;
}

// add dummy keys \

void dummy_keys (unsigned int *tmpRelR, unsigned int *tmpRelS){

	// -- to be added
	for(uint k = 1; k <= partition_num; k ++){

		for(uint i = k * one_r_partition_num - 1; i >= k * one_r_partition_num - 8; i --) {
			tmpRelR[2*i] = ((i) & 0xf) << NUM_RADIX_BITS;
			tmpRelR[2*i + 1] = 0xffffffff;
		}
	}

	for(uint k = 1; k <= partition_num; k ++){

		for(uint i = k * one_s_partition_num - 1; i >= k * one_s_partition_num - 8; i --) {
			//tmpRelS[2*i] = ((i) & 0xf);
			tmpRelS[2*i + 1] = 0xffffffff;
			//printf("%d\n",2*i);

		}
	}
}

// serial partition using CPU
int64_t RJonCPU( tuple_t * rTable, tuple_t * sTable, uint rTupleNum, uint sTupleNum){
	relation_t  relR;
	relation_t  relS;
	int64_t result = 0;

	relR.tuples = rTable;
	relR.num_tuples = rTupleNum;

	relS.tuples = sTable;
	relS.num_tuples = sTupleNum;

	result = RJ(&relR, &relS, 1);

	return result;
}
// partition using multi-thread

int64_t parallel_partition( 
		tuple_t * rTable, tuple_t * sTable, uint rTupleNum, uint sTupleNum, 
		tuple_t * tmpRelR, tuple_t * tmpRelS, int nthreads
		){
	relation_t  relR;
	relation_t  relS;
	int64_t result = 0;

	relR.tuples = rTable;
	relR.num_tuples = rTupleNum;

	relS.tuples = sTable;
	relS.num_tuples = sTupleNum;

	result = join_init_run(&relR, &relS, tmpRelR, tmpRelS, nthreads);

	return result;
}

int64_t RJonFPGA( tuple_t * rTable, tuple_t * sTable, uint rTupleNum, uint sTupleNum){
	relation_t  relR;
	relation_t  relS;
	int64_t result = 0;

	relR.tuples = rTable;
	relR.num_tuples = rTupleNum;

	relS.tuples = sTable;
	relS.num_tuples = sTupleNum;

	result = RJ_FPGA(&relR, &relS, 1);

	return result;
}

//static const size_t workSize = 128*1;

int main(int argc, char *argv[]) {
	cl_uint num_platforms;
	cl_uint num_devices;
	status = setHardwareEnv(num_platforms, num_devices);
	printf("Num of radix bits = %d\n", NUM_RADIX_BITS);
	printf("S R Ratio = %d\n", srRatio);
	printf("Creating host buffers.\n");
	sTableReadRange = (unsigned int *) malloc( sizeof(unsigned int) * 2);
	rTableReadRange = (unsigned int *) malloc( sizeof(unsigned int) * 2);

	// create memory space for R & S tables
#if PLATFORM == HARP
	rTable = (unsigned int*) malloc (sizeof (unsigned int) * rTableSize);
	sTable = (unsigned int*) malloc (sizeof (unsigned int) * sTableSize);
	matchedTable = (unsigned int*)clSVMAllocAltera(context, 0, matchedTableSize, 1024);
#elif PLATFORM == DE5
	posix_memalign ((void **)&rTable, AOCL_ALIGNMENT, rTableSize);
	posix_memalign ((void **)&sTable, AOCL_ALIGNMENT, sTableSize);
	posix_memalign ((void **)&matchedTable, AOCL_ALIGNMENT, matchedTableSize);
	//HashTable = (unsigned int*)clSVMAllocAltera(context,0,sizeof(unsigned int)*2*hashBucketSize * rHashTableBucketNum,1024);
#endif

	if(!rTable || !sTable || !matchedTable) {
		dump_error("Failed to allocate buffers.", status);
		freeResources();
		return 1;	 
	}
	// create memory space for partitioned tables
	tuple_t * tmpRelR;            
	tuple_t * tmpRelS;
	
#if PLATFORM == HARP
	tmpRelR = (tuple_t*)clSVMAllocAltera(context, 0, rTableSize, 1024); 
	tmpRelS = (tuple_t*)clSVMAllocAltera(context, 0, sTableSize, 1024);
#elif PLATFORM == DE5
	posix_memalign ((void **)&tmpRelR, AOCL_ALIGNMENT, rTableSize);
	posix_memalign ((void **)&tmpRelS, AOCL_ALIGNMENT, sTableSize);
#endif

	if (!tmpRelR || !tmpRelS) {
		dump_error("Failed to allocate buffers (tmpRelR or tmpRelS).", status);
		freeResources();
		return 1;
	}

	// tuple_t * tmpRelR = (tuple_t*) alloc_aligned(rTupleNum * sizeof(tuple_t) +
	//                                       RELATION_PADDING);
	// tuple_t * tmpRelS = (tuple_t*) alloc_aligned(sTupleNum * sizeof(tuple_t) +
	//                                       RELATION_PADDING);
	printf("tmpRelR size %d MB \n", (rTupleNum * sizeof(tuple_t) + RELATION_PADDING)/1024/1024);
	printf("tmpRelS size %d MB \n", (sTupleNum * sizeof(tuple_t) + RELATION_PADDING)/1024/1024);

	// create dataset 
	create_relation_pk(rTable, rTupleNum);
	create_relation_pk(sTable, sTupleNum);
	memset (matchedTable,0,matchedTableSize);
	printf("create dataset\n");

	//	partitioning in parallel - multi-thread version
	const double partition_start_time = getCurrentTimestamp();
	parallel_partition( (tuple_t *) rTable, (tuple_t * )sTable,  rTupleNum,  sTupleNum, (tuple_t *) tmpRelR, (tuple_t * )tmpRelS, nthreads);
	const double partition_end_time = getCurrentTimestamp();
	printf("[INFO] Partitioning time is: %f ms\n", (partition_end_time - partition_start_time)*1000);

#if 0
	// add dummy keys after partitioning. can also do before partitioning.
	dummy_keys((unsigned int *)(tmpRelR), (unsigned int *)(tmpRelS));

	// for (int i  = 0; i < 100; i ++)
	// 	printf("%d \t", rTable[2*i]);
	// printf("\n");
	// printf("\n");
	// for (int i  = 0; i < 100; i ++)
	// 	printf("%d \t", tmpRelR[2*i]);
	// printf("\n");
	// for (int i  = 0; i < 100; i ++)
	// 	printf("%d \t", tmpRelS[2*i]);
	// printf("\n");

	//-------Init the hardware env------//
	setKernelEnv();
	printf("Creating hash join kernel\n");		
	creatKernels();
	//---------------------------------//

#if PLATFORM == DE5
	cl_mem rTableOnDevice = clCreateBuffer(context,CL_MEM_READ_WRITE,rTableSize,0,&status);
	if (status != CL_SUCCESS){
		cout << "Create rTableOnDevice failed" << endl;
		exit(1);
	}
	cl_mem sTableOnDevice = clCreateBuffer(context,CL_MEM_READ_WRITE,sTableSize,0,&status);
	if (status != CL_SUCCESS){
		cout << "Create sTableOnDevice failed" << endl;
		exit(1);
	}
	cl_mem matchedTableOnDevice = clCreateBuffer(context,CL_MEM_READ_WRITE,matchedTableSize,0,&status);
	if (status != CL_SUCCESS){
		cout << "Create matchedTableOnDevice failed" << endl;
		exit(1);
	}

	const double copy_start_time = getCurrentTimestamp();

	//write partitioned tables on host into device buffer
	status = clEnqueueWriteBuffer(queue_relRead,rTableOnDevice,CL_TRUE,0,rTableSize,tmpRelR,0,NULL,NULL);
	if (status != CL_SUCCESS){
		cout << "writing  R table into buffer failed" << endl;
		exit(1);
	}
	status = clEnqueueWriteBuffer(queue_relRead,sTableOnDevice,CL_TRUE,0,sTableSize,tmpRelS,0,NULL,NULL);
	if (status != CL_SUCCESS){
		cout << "writing  S table into buffer failed" << endl;
		exit(1);
	}
	status = clEnqueueWriteBuffer(queue_hashjoin,matchedTableOnDevice,CL_TRUE,0,matchedTableSize,matchedTable,0,NULL,NULL);
	if (status != CL_SUCCESS){
		cout << "writing matchedTable table into buffer failed" << endl;
		exit(1);
	}
	// printout the PCIE copy time
	const double copy_end_time = getCurrentTimestamp();
	printf("[INFO] copy to device time is: %f ms\n", (copy_end_time - copy_start_time)*1000);
#endif

#if PLATFORM == CPU
	// // ------------------ Process on CPU -------------------------//
	// 	// single thread partitioning then join
	const double cpu_start_time = getCurrentTimestamp();
	int64_t result = RJonCPU( (tuple_t *)rTable, (tuple_t *)sTable, rTupleNum, sTupleNum);
	const double cpu_end_time = getCurrentTimestamp();
	printf("[INFO] cpu processing result is %lu \n", result);
	printf("[INFO] cpu runtime is: %f\n", (cpu_end_time - cpu_start_time)*1000);
	// //-------------------------------------------------------------//
#endif

	int argvi = 0;
	rTableReadRange[0] = 0;
	rTableReadRange[1] = rTupleNum;
	sTableReadRange[0] = 0;
	sTableReadRange[1] = sTupleNum;    
	//set the arguments
	uint cur_part = 0;

#if PLATFORM == HARP
	uint r_offset = 0;
	uint s_offset = 0;

	argvi = 0;
	clSetKernelArgSVMPointerAltera(kernel_relRead, argvi ++, (void*)tmpRelR);
	clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_uint),(void*)&one_r_partition_num);
	clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_uint),(void*)&r_offset);

	clSetKernelArgSVMPointerAltera(kernel_relRead, argvi ++, (void*)tmpRelS);
	clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_uint),(void*)&one_s_partition_num);
	clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_uint),(void*)&s_offset);

	clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_uint),(void*)&inp_num_radix_bits);

	argvi = 0;
	clSetKernelArgSVMPointerAltera(kernel_hashjoin, argvi ++, (void*)matchedTable);
#elif PLATFORM == DE5
	argvi = 0;
	clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_mem),&rTableOnDevice);
	// clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_uint),(void*)&rTupleNum);
	clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_uint),(void*)&one_r_partition_num);
	clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_mem),&sTableOnDevice);
	// clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_uint),(void*)&sTupleNum);
	clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_uint),(void*)&one_s_partition_num);
	#if RB_DEPENDENT == 0
		clSetKernelArg(kernel_relRead,argvi ++,sizeof(cl_uint),(void*)&inp_num_radix_bits);
	#endif

	argvi = 0;
	clSetKernelArg(kernel_hashjoin,argvi ++,sizeof(cl_mem),&matchedTableOnDevice);
	//clSetKernelArg(kernel_hashjoin,argvi ++,sizeof(cl_uint),(void*)&cur_part);
	//clSetKernelArg(kernel_hashjoin,argvi ++,sizeof(cl_uint),(void*)&sTupleNum);
#endif

	printf("Launching the kernels...\n");

	cl_event event_hashjoin, event_relRead, event_gather;
	cl_event event_filter[NUM_FPGA_DATAPATH];
	//cl_event event_sfilter[NUM_FPGA_DATAPATH];
	//cl_event event_sgather;
	//status = clEnqueueNDRangeKernel(queue,kernel,1,NULL,&gworkSize_build,&workSize,0,NULL,&event1);

	status = clEnqueueTask(queue_gather, kernel_gather, 0, NULL, &event_gather);
	if (status != CL_SUCCESS) {
		dump_error("Failed to launch kernel 2 .", status);
		freeResources();
		return 1;
	}

/*
	status = clEnqueueTask(queue_sgather, kernel_sgather, 0, NULL, &event_sgather);
	if (status != CL_SUCCESS) {
		dump_error("Failed to launch kernel 3.", status);
		freeResources();
		return 1;
	}
*/

	for(int i = 0; i < NUM_FPGA_DATAPATH; i++){
		status = clEnqueueTask(queue_filter[i], kernel_filter[i], 0, NULL, &event_filter[i]);
		if (status != CL_SUCCESS) {
			dump_error("Failed to launch kernel 4.", status);
			freeResources();
			return 1;
		}
	}

	printf("[INFO] Begin processing on FPGA\n");
	const double start_time = getCurrentTimestamp();
	for(int i = 0; i < partition_num_loopcount; i ++){	
		const double start_enqueue_time = getCurrentTimestamp();

		status = clEnqueueTask(queue_relRead, kernel_relRead, 0, NULL, &event_relRead);

		status = clEnqueueTask(queue_hashjoin, kernel_hashjoin, 0, NULL, &event_hashjoin);
		
		const double end_enqueue_time = getCurrentTimestamp();
		
		clFinish(queue_hashjoin);

		r_offset += one_r_partition_num;
		s_offset += one_s_partition_num;
		//const double end_time = getCurrentTimestamp();
		double enqueue_time = (end_enqueue_time - start_enqueue_time);
		//double time = (end_time - start_enqueue_time);
		//printf("[INFO] FPGA runtime is: %f\n", time * 1000);
		//printf("[INFO] Enqueue time is: %f ms\n", enqueue_time * 1000);
	}		
	const double end_time = getCurrentTimestamp();
	double time = (end_time - start_time);
	printf("[INFO] FPGA runtime is: %f ms\n", time * 1000);
	printf("[INFO] Throughput is %f MTEPS\n", rTupleNum * (1+srRatio) / time /1000000);

#if PLATFORM == DE5
	// read the results back 
	status = clEnqueueReadBuffer(queue_hashjoin,matchedTableOnDevice,CL_TRUE,0,matchedTableSize,matchedTable,0,NULL,NULL);
#endif

	//for (int i  = 0; i < 100; i ++)
	//	printf("%d \t", matchedTable[i]);
	//printf("\n");
	printf("No of matches - %d \n", matchedTable[0]);

	freeResources();
#endif
	return 0;
}
