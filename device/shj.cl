/* simple hash join on BRAM */
//----channel define----//
typedef struct shuffledData{
  uint num;
  uint idx;
  } shuffled_type;

typedef struct filterData{
  bool end;
  uchar num;
  uint2 data[8];
  } filter_type;

typedef struct RtableData{
  uchar hash_val[8];
  uint16 data;
} Rtuples_type;


#define DEBUG 0

#define ENDFLAG 0xffffffff

//#define SHIFT_AT_START 0
//#define RB_DEPENDENT 0

#define NUM_FPGA_DATAPATH 8
#define NUM_FPGA_DATAPATH_BITS 3

//#define NUM_PIPELINES 16
//#define NUM_PIPELINES_BITS 4
//#define NUM_RADIX_BITS 7

#define HASH(K, MASK, SKIP) (((K) & MASK) >> SKIP)
#define RELR_L_NUM 1024*256
#define HASHTABLE_L_SIZE 1024*256

//#if NUM_FPGA_DATAPATH == 16
//	#define HASHTABLE_BUCKET_SIZE 4 // 2 byte per 
//#else
#define HASHTABLE_BUCKET_SIZE 1 // 2 byte per 
//#endif

#define HASHTABLE_BUCKET_NUM HASHTABLE_L_SIZE/HASHTABLE_BUCKET_SIZE
//#define TRUNC_BITS RELR_L_NUM/NUM_PIPELINES/HASHTABLE_BUCKET_SIZE-1


channel Rtuples_type relR[NUM_FPGA_DATAPATH] __attribute__((depth(128)));
channel uint2 relS[NUM_FPGA_DATAPATH][NUM_FPGA_DATAPATH] __attribute__((depth(128)));

//channel uint gatherFlagCh[NUM_FPGA_DATAPATH] __attribute__((depth(64)));
//channel uint filterFlagCh[NUM_FPGA_DATAPATH] __attribute__((depth(64)));
channel uint2 buildCh[NUM_FPGA_DATAPATH] __attribute__((depth(512)));
channel filter_type toFilterCh[NUM_FPGA_DATAPATH] __attribute__((depth(512)));
//channel filter_type toFilterCh[NUM_FPGA_DATAPATH] __attribute__((depth(128)));

//channel uint s_gatherFlagCh[NUM_FPGA_DATAPATH] __attribute__((depth(64)));
//channel uint s_filterFlagCh[NUM_FPGA_DATAPATH] __attribute__((depth(64)));
//channel uint2 s_buildCh[NUM_FPGA_DATAPATH] __attribute__((depth(512)));
//channel filter_type s_toFilterCh[NUM_FPGA_DATAPATH] __attribute__((depth(512)));
channel uint s_endFlagCh __attribute__((depth(64)));

//channel uint start_channel __attribute__((depth(8)));


//#define SW
	__attribute__((task))
__kernel void relRead (
		__global uint16 * restrict rTable, 
		const uint rTuples,
		const uint rOffset,
		__global uint16 * restrict sTable, 
		const uint sTuples,
		const uint sOffset, 
		const uint inp_num_rb
		)
{
	//uint rTableReadNum = rTuples;
	//uint sTableReadNum = sTuples;

	uint rTableOffset = rOffset;
	uint rPartitionSize = rTuples;

	uint sPartitionSize = sTuples;
	uint sTableOffset = sOffset;


#if NUM_FPGA_DATAPATH > 8
	char num_fpga_datapath_offset = 15;
#elif NUM_FPGA_DATAPATH > 4
	char num_fpga_datapath_offset = 7;
#else
	char num_fpga_datapath_offset = 3;
#endif

	//char num_fpga_datapath_offset = NUM_FPGA_DATAPATH-1;

#if DEBUG
	int counter = 0;
#endif

//	while(true){
//		bool valid;
//		uint flga_data;
//		uint offset = read_channel_nb_altera(start_channel, &valid);
//		if (valid){

		for(int i = (rTableOffset) >> 3; i < (rTableOffset + rPartitionSize) >>3; i ++){
			uint16 rtable_uint16 = rTable[i];

			Rtuples_type tuples;

			uint16 rtable_shifted_uint16;

			rtable_shifted_uint16.s0 = rtable_uint16.s0 >> inp_num_rb;
			rtable_shifted_uint16.s1 = rtable_uint16.s1;
			rtable_shifted_uint16.s2 = rtable_uint16.s2 >> inp_num_rb;
			rtable_shifted_uint16.s3 = rtable_uint16.s3;
			rtable_shifted_uint16.s4 = rtable_uint16.s4 >> inp_num_rb;
			rtable_shifted_uint16.s5 = rtable_uint16.s5;
			rtable_shifted_uint16.s6 = rtable_uint16.s6 >> inp_num_rb;
			rtable_shifted_uint16.s7 = rtable_uint16.s7;
			rtable_shifted_uint16.s8 = rtable_uint16.s8 >> inp_num_rb;
			rtable_shifted_uint16.s9 = rtable_uint16.s9;
			rtable_shifted_uint16.sa = rtable_uint16.sa >> inp_num_rb;
			rtable_shifted_uint16.sb = rtable_uint16.sb;
			rtable_shifted_uint16.sc = rtable_uint16.sc >> inp_num_rb;
			rtable_shifted_uint16.sd = rtable_uint16.sd;
			rtable_shifted_uint16.se = rtable_uint16.se >> inp_num_rb;
			rtable_shifted_uint16.sf = rtable_uint16.sf;
/*
			rtable_shifted_uint16.s0 = rtable_uint16.s0;
			rtable_shifted_uint16.s1 = rtable_uint16.s1;
			rtable_shifted_uint16.s2 = rtable_uint16.s2;
			rtable_shifted_uint16.s3 = rtable_uint16.s3;
			rtable_shifted_uint16.s4 = rtable_uint16.s4;
			rtable_shifted_uint16.s5 = rtable_uint16.s5;
			rtable_shifted_uint16.s6 = rtable_uint16.s6;
			rtable_shifted_uint16.s7 = rtable_uint16.s7;
			rtable_shifted_uint16.s8 = rtable_uint16.s8;
			rtable_shifted_uint16.s9 = rtable_uint16.s9;
			rtable_shifted_uint16.sa = rtable_uint16.sa;
			rtable_shifted_uint16.sb = rtable_uint16.sb;
			rtable_shifted_uint16.sc = rtable_uint16.sc;
			rtable_shifted_uint16.sd = rtable_uint16.sd;
			rtable_shifted_uint16.se = rtable_uint16.se;
			rtable_shifted_uint16.sf = rtable_uint16.sf;
*/
			tuples.data = rtable_shifted_uint16;

			tuples.hash_val[0] = (rtable_shifted_uint16.s0 & num_fpga_datapath_offset);
			tuples.hash_val[1] = (rtable_shifted_uint16.s2 & num_fpga_datapath_offset);
			tuples.hash_val[2] = (rtable_shifted_uint16.s4 & num_fpga_datapath_offset);
			tuples.hash_val[3] = (rtable_shifted_uint16.s6 & num_fpga_datapath_offset);
			tuples.hash_val[4] = (rtable_shifted_uint16.s8 & num_fpga_datapath_offset);
			tuples.hash_val[5] = (rtable_shifted_uint16.sa & num_fpga_datapath_offset);
			tuples.hash_val[6] = (rtable_shifted_uint16.sc & num_fpga_datapath_offset);
			tuples.hash_val[7] = (rtable_shifted_uint16.se & num_fpga_datapath_offset);

			#pragma unroll NUM_FPGA_DATAPATH
			for (int j = 0; j < NUM_FPGA_DATAPATH; j++) {
				write_channel_altera(relR[j], tuples);
			}

			#if DEBUG
				//printf("i = %d\n", i);
				if (i == 32767) {
					printf("Before shift\n");
					printf("r[0].key = %d, r[0].val = %d\n", rtable_uint16.s0, rtable_uint16.s1);
					printf("r[1].key = %d, r[1].val = %d\n", rtable_uint16.s2, rtable_uint16.s3);
					printf("r[2].key = %d, r[2].val = %d\n", rtable_uint16.s4, rtable_uint16.s5);
					printf("r[3].key = %d, r[3].val = %d\n", rtable_uint16.s6, rtable_uint16.s7);
					printf("r[4].key = %d, r[4].val = %d\n", rtable_uint16.s8, rtable_uint16.s9);
					printf("r[5].key = %d, r[5].val = %d\n", rtable_uint16.sa, rtable_uint16.sb);
					printf("r[6].key = %d, r[6].val = %d\n", rtable_uint16.sc, rtable_uint16.sd);
					printf("r[7].key = %d, r[7].val = %d\n", rtable_uint16.se, rtable_uint16.sf);

					printf("After shift\n");
					printf("r[0].key = %d, r[0].val = %d\n", rtable_shifted_uint16.s0, rtable_shifted_uint16.s1);
					printf("r[1].key = %d, r[1].val = %d\n", rtable_shifted_uint16.s2, rtable_shifted_uint16.s3);
					printf("r[2].key = %d, r[2].val = %d\n", rtable_shifted_uint16.s4, rtable_shifted_uint16.s5);
					printf("r[3].key = %d, r[3].val = %d\n", rtable_shifted_uint16.s6, rtable_shifted_uint16.s7);
					printf("r[4].key = %d, r[4].val = %d\n", rtable_shifted_uint16.s8, rtable_shifted_uint16.s9);
					printf("r[5].key = %d, r[5].val = %d\n", rtable_shifted_uint16.sa, rtable_shifted_uint16.sb);
					printf("r[6].key = %d, r[6].val = %d\n", rtable_shifted_uint16.sc, rtable_shifted_uint16.sd);
					printf("r[7].key = %d, r[7].val = %d\n", rtable_shifted_uint16.se, rtable_shifted_uint16.sf);

				}
			#endif
		}

		for(int i = sTableOffset >> 3; i < (sTableOffset + sPartitionSize) >> 3; i ++){
			uint16 stable_uint16 = sTable[i];

			uint2 stable_shifted[8];

			stable_shifted[0].x = stable_uint16.s0 >> inp_num_rb;
			stable_shifted[0].y = stable_uint16.s1;
			stable_shifted[1].x = stable_uint16.s2 >> inp_num_rb;
			stable_shifted[1].y = stable_uint16.s3;
			stable_shifted[2].x = stable_uint16.s4 >> inp_num_rb;
			stable_shifted[2].y = stable_uint16.s5;
			stable_shifted[3].x = stable_uint16.s6 >> inp_num_rb;
			stable_shifted[3].y = stable_uint16.s7;
			stable_shifted[4].x = stable_uint16.s8 >> inp_num_rb;
			stable_shifted[4].y = stable_uint16.s9;
			stable_shifted[5].x = stable_uint16.sa >> inp_num_rb;
			stable_shifted[5].y = stable_uint16.sb;
			stable_shifted[6].x = stable_uint16.sc >> inp_num_rb;
			stable_shifted[6].y = stable_uint16.sd;
			stable_shifted[7].x = stable_uint16.se >> inp_num_rb;
			stable_shifted[7].y = stable_uint16.sf;

			#pragma unroll NUM_FPGA_DATAPATH
			for (int j = 0; j < NUM_FPGA_DATAPATH; j++) {
				switch (stable_shifted[j].x & num_fpga_datapath_offset) {
					case 0: write_channel_altera(relS[j][0], stable_shifted[j]); break;
					case 1: write_channel_altera(relS[j][1], stable_shifted[j]); break;
					case 2: write_channel_altera(relS[j][2], stable_shifted[j]); break;
					case 3: write_channel_altera(relS[j][3], stable_shifted[j]); break;
					case 4: write_channel_altera(relS[j][4], stable_shifted[j]); break;
					case 5: write_channel_altera(relS[j][5], stable_shifted[j]); break;
					case 6: write_channel_altera(relS[j][6], stable_shifted[j]); break;
					case 7: write_channel_altera(relS[j][7], stable_shifted[j]); break;
				}
			}
//				if (stable_uint16.s1 == ENDFLAG) {
//					#if DEBUG
//						printf("Found ENDFLAG\n");
//					#endif
//				}
		}
		
		write_channel_altera(s_endFlagCh, ENDFLAG);

	#if DEBUG
		counter++;
		printf("counter = %d\n", counter);
	#endif

}


__attribute__((always_inline)) shuffled_type decoder(uchar opcode){
	uint idx;
	uint num;
	switch(opcode){
		case 0: idx = 0; num = 0; break;
		case 1: idx = 0; num = 1; break;
		case 2: idx = 1; num = 1; break;
		case 3: idx = 8; num = 2; break;
		case 4: idx = 2; num = 1; break;
		case 5: idx = 16; num = 2; break;
		case 6: idx = 17; num = 2; break;
		case 7: idx = 136; num = 3; break;
		case 8: idx = 3; num = 1; break;
		case 9: idx = 24; num = 2; break;
		case 10: idx = 25; num = 2; break;
		case 11: idx = 200; num = 3; break;
		case 12: idx = 26; num = 2; break;
		case 13: idx = 208; num = 3; break;
		case 14: idx = 209; num = 3; break;
		case 15: idx = 1672; num = 4; break;
		case 16: idx = 4; num = 1; break;
		case 17: idx = 32; num = 2; break;
		case 18: idx = 33; num = 2; break;
		case 19: idx = 264; num = 3; break;
		case 20: idx = 34; num = 2; break;
		case 21: idx = 272; num = 3; break;
		case 22: idx = 273; num = 3; break;
		case 23: idx = 2184; num = 4; break;
		case 24: idx = 35; num = 2; break;
		case 25: idx = 280; num = 3; break;
		case 26: idx = 281; num = 3; break;
		case 27: idx = 2248; num = 4; break;
		case 28: idx = 282; num = 3; break;
		case 29: idx = 2256; num = 4; break;
		case 30: idx = 2257; num = 4; break;
		case 31: idx = 18056; num = 5; break;
		case 32: idx = 5; num = 1; break;
		case 33: idx = 40; num = 2; break;
		case 34: idx = 41; num = 2; break;
		case 35: idx = 328; num = 3; break;
		case 36: idx = 42; num = 2; break;
		case 37: idx = 336; num = 3; break;
		case 38: idx = 337; num = 3; break;
		case 39: idx = 2696; num = 4; break;
		case 40: idx = 43; num = 2; break;
		case 41: idx = 344; num = 3; break;
		case 42: idx = 345; num = 3; break;
		case 43: idx = 2760; num = 4; break;
		case 44: idx = 346; num = 3; break;
		case 45: idx = 2768; num = 4; break;
		case 46: idx = 2769; num = 4; break;
		case 47: idx = 22152; num = 5; break;
		case 48: idx = 44; num = 2; break;
		case 49: idx = 352; num = 3; break;
		case 50: idx = 353; num = 3; break;
		case 51: idx = 2824; num = 4; break;
		case 52: idx = 354; num = 3; break;
		case 53: idx = 2832; num = 4; break;
		case 54: idx = 2833; num = 4; break;
		case 55: idx = 22664; num = 5; break;
		case 56: idx = 355; num = 3; break;
		case 57: idx = 2840; num = 4; break;
		case 58: idx = 2841; num = 4; break;
		case 59: idx = 22728; num = 5; break;
		case 60: idx = 2842; num = 4; break;
		case 61: idx = 22736; num = 5; break;
		case 62: idx = 22737; num = 5; break;
		case 63: idx = 181896; num = 6; break;
		case 64: idx = 6; num = 1; break;
		case 65: idx = 48; num = 2; break;
		case 66: idx = 49; num = 2; break;
		case 67: idx = 392; num = 3; break;
		case 68: idx = 50; num = 2; break;
		case 69: idx = 400; num = 3; break;
		case 70: idx = 401; num = 3; break;
		case 71: idx = 3208; num = 4; break;
		case 72: idx = 51; num = 2; break;
		case 73: idx = 408; num = 3; break;
		case 74: idx = 409; num = 3; break;
		case 75: idx = 3272; num = 4; break;
		case 76: idx = 410; num = 3; break;
		case 77: idx = 3280; num = 4; break;
		case 78: idx = 3281; num = 4; break;
		case 79: idx = 26248; num = 5; break;
		case 80: idx = 52; num = 2; break;
		case 81: idx = 416; num = 3; break;
		case 82: idx = 417; num = 3; break;
		case 83: idx = 3336; num = 4; break;
		case 84: idx = 418; num = 3; break;
		case 85: idx = 3344; num = 4; break;
		case 86: idx = 3345; num = 4; break;
		case 87: idx = 26760; num = 5; break;
		case 88: idx = 419; num = 3; break;
		case 89: idx = 3352; num = 4; break;
		case 90: idx = 3353; num = 4; break;
		case 91: idx = 26824; num = 5; break;
		case 92: idx = 3354; num = 4; break;
		case 93: idx = 26832; num = 5; break;
		case 94: idx = 26833; num = 5; break;
		case 95: idx = 214664; num = 6; break;
		case 96: idx = 53; num = 2; break;
		case 97: idx = 424; num = 3; break;
		case 98: idx = 425; num = 3; break;
		case 99: idx = 3400; num = 4; break;
		case 100: idx = 426; num = 3; break;
		case 101: idx = 3408; num = 4; break;
		case 102: idx = 3409; num = 4; break;
		case 103: idx = 27272; num = 5; break;
		case 104: idx = 427; num = 3; break;
		case 105: idx = 3416; num = 4; break;
		case 106: idx = 3417; num = 4; break;
		case 107: idx = 27336; num = 5; break;
		case 108: idx = 3418; num = 4; break;
		case 109: idx = 27344; num = 5; break;
		case 110: idx = 27345; num = 5; break;
		case 111: idx = 218760; num = 6; break;
		case 112: idx = 428; num = 3; break;
		case 113: idx = 3424; num = 4; break;
		case 114: idx = 3425; num = 4; break;
		case 115: idx = 27400; num = 5; break;
		case 116: idx = 3426; num = 4; break;
		case 117: idx = 27408; num = 5; break;
		case 118: idx = 27409; num = 5; break;
		case 119: idx = 219272; num = 6; break;
		case 120: idx = 3427; num = 4; break;
		case 121: idx = 27416; num = 5; break;
		case 122: idx = 27417; num = 5; break;
		case 123: idx = 219336; num = 6; break;
		case 124: idx = 27418; num = 5; break;
		case 125: idx = 219344; num = 6; break;
		case 126: idx = 219345; num = 6; break;
		case 127: idx = 1754760; num = 7; break;
		case 128: idx = 7; num = 1; break;
		case 129: idx = 56; num = 2; break;
		case 130: idx = 57; num = 2; break;
		case 131: idx = 456; num = 3; break;
		case 132: idx = 58; num = 2; break;
		case 133: idx = 464; num = 3; break;
		case 134: idx = 465; num = 3; break;
		case 135: idx = 3720; num = 4; break;
		case 136: idx = 59; num = 2; break;
		case 137: idx = 472; num = 3; break;
		case 138: idx = 473; num = 3; break;
		case 139: idx = 3784; num = 4; break;
		case 140: idx = 474; num = 3; break;
		case 141: idx = 3792; num = 4; break;
		case 142: idx = 3793; num = 4; break;
		case 143: idx = 30344; num = 5; break;
		case 144: idx = 60; num = 2; break;
		case 145: idx = 480; num = 3; break;
		case 146: idx = 481; num = 3; break;
		case 147: idx = 3848; num = 4; break;
		case 148: idx = 482; num = 3; break;
		case 149: idx = 3856; num = 4; break;
		case 150: idx = 3857; num = 4; break;
		case 151: idx = 30856; num = 5; break;
		case 152: idx = 483; num = 3; break;
		case 153: idx = 3864; num = 4; break;
		case 154: idx = 3865; num = 4; break;
		case 155: idx = 30920; num = 5; break;
		case 156: idx = 3866; num = 4; break;
		case 157: idx = 30928; num = 5; break;
		case 158: idx = 30929; num = 5; break;
		case 159: idx = 247432; num = 6; break;
		case 160: idx = 61; num = 2; break;
		case 161: idx = 488; num = 3; break;
		case 162: idx = 489; num = 3; break;
		case 163: idx = 3912; num = 4; break;
		case 164: idx = 490; num = 3; break;
		case 165: idx = 3920; num = 4; break;
		case 166: idx = 3921; num = 4; break;
		case 167: idx = 31368; num = 5; break;
		case 168: idx = 491; num = 3; break;
		case 169: idx = 3928; num = 4; break;
		case 170: idx = 3929; num = 4; break;
		case 171: idx = 31432; num = 5; break;
		case 172: idx = 3930; num = 4; break;
		case 173: idx = 31440; num = 5; break;
		case 174: idx = 31441; num = 5; break;
		case 175: idx = 251528; num = 6; break;
		case 176: idx = 492; num = 3; break;
		case 177: idx = 3936; num = 4; break;
		case 178: idx = 3937; num = 4; break;
		case 179: idx = 31496; num = 5; break;
		case 180: idx = 3938; num = 4; break;
		case 181: idx = 31504; num = 5; break;
		case 182: idx = 31505; num = 5; break;
		case 183: idx = 252040; num = 6; break;
		case 184: idx = 3939; num = 4; break;
		case 185: idx = 31512; num = 5; break;
		case 186: idx = 31513; num = 5; break;
		case 187: idx = 252104; num = 6; break;
		case 188: idx = 31514; num = 5; break;
		case 189: idx = 252112; num = 6; break;
		case 190: idx = 252113; num = 6; break;
		case 191: idx = 2016904; num = 7; break;
		case 192: idx = 62; num = 2; break;
		case 193: idx = 496; num = 3; break;
		case 194: idx = 497; num = 3; break;
		case 195: idx = 3976; num = 4; break;
		case 196: idx = 498; num = 3; break;
		case 197: idx = 3984; num = 4; break;
		case 198: idx = 3985; num = 4; break;
		case 199: idx = 31880; num = 5; break;
		case 200: idx = 499; num = 3; break;
		case 201: idx = 3992; num = 4; break;
		case 202: idx = 3993; num = 4; break;
		case 203: idx = 31944; num = 5; break;
		case 204: idx = 3994; num = 4; break;
		case 205: idx = 31952; num = 5; break;
		case 206: idx = 31953; num = 5; break;
		case 207: idx = 255624; num = 6; break;
		case 208: idx = 500; num = 3; break;
		case 209: idx = 4000; num = 4; break;
		case 210: idx = 4001; num = 4; break;
		case 211: idx = 32008; num = 5; break;
		case 212: idx = 4002; num = 4; break;
		case 213: idx = 32016; num = 5; break;
		case 214: idx = 32017; num = 5; break;
		case 215: idx = 256136; num = 6; break;
		case 216: idx = 4003; num = 4; break;
		case 217: idx = 32024; num = 5; break;
		case 218: idx = 32025; num = 5; break;
		case 219: idx = 256200; num = 6; break;
		case 220: idx = 32026; num = 5; break;
		case 221: idx = 256208; num = 6; break;
		case 222: idx = 256209; num = 6; break;
		case 223: idx = 2049672; num = 7; break;
		case 224: idx = 501; num = 3; break;
		case 225: idx = 4008; num = 4; break;
		case 226: idx = 4009; num = 4; break;
		case 227: idx = 32072; num = 5; break;
		case 228: idx = 4010; num = 4; break;
		case 229: idx = 32080; num = 5; break;
		case 230: idx = 32081; num = 5; break;
		case 231: idx = 256648; num = 6; break;
		case 232: idx = 4011; num = 4; break;
		case 233: idx = 32088; num = 5; break;
		case 234: idx = 32089; num = 5; break;
		case 235: idx = 256712; num = 6; break;
		case 236: idx = 32090; num = 5; break;
		case 237: idx = 256720; num = 6; break;
		case 238: idx = 256721; num = 6; break;
		case 239: idx = 2053768; num = 7; break;
		case 240: idx = 4012; num = 4; break;
		case 241: idx = 32096; num = 5; break;
		case 242: idx = 32097; num = 5; break;
		case 243: idx = 256776; num = 6; break;
		case 244: idx = 32098; num = 5; break;
		case 245: idx = 256784; num = 6; break;
		case 246: idx = 256785; num = 6; break;
		case 247: idx = 2054280; num = 7; break;
		case 248: idx = 32099; num = 5; break;
		case 249: idx = 256792; num = 6; break;
		case 250: idx = 256793; num = 6; break;
		case 251: idx = 2054344; num = 7; break;
		case 252: idx = 256794; num = 6; break;
		case 253: idx = 2054352; num = 7; break;
		case 254: idx = 2054353; num = 7; break;
		case 255: idx = 16434824; num = 8; break;
		default: idx = 0; num = 0; break;
	}
	shuffled_type data;
	data.idx = idx;
	data.num = num;
	return data;
}

__attribute__((task))
__kernel void gather ()                
{
	bool engine_finish[NUM_FPGA_DATAPATH];

	#pragma unroll NUM_FPGA_DATAPATH
	for(int j = 0; j < NUM_FPGA_DATAPATH; j ++)
		engine_finish[j] = false;

	while(true){
		#pragma unroll NUM_FPGA_DATAPATH
		for(int i = 0; i < NUM_FPGA_DATAPATH; i ++){ 
			// each collect engine do their work

			uint16 data_r;
			bool valid_c;
			bool valid_r[8];
			uchar idx[8];


			#pragma unroll 8
			for(int i = 0; i < 8; i ++){
				valid_r[i] = false;
			}

			#pragma unroll 8
			for(int i = 0; i < 8; i ++){
				idx[i] = false;
			}

			Rtuples_type tuples = read_channel_altera(relR[i]);

			data_r = tuples.data;

			valid_r[0] = tuples.hash_val[0] == i ? 1:0;
			valid_r[1] = tuples.hash_val[1] == i ? 1:0;
			valid_r[2] = tuples.hash_val[2] == i ? 1:0;
			valid_r[3] = tuples.hash_val[3] == i ? 1:0;
			valid_r[4] = tuples.hash_val[4] == i ? 1:0;
			valid_r[5] = tuples.hash_val[5] == i ? 1:0;
			valid_r[6] = tuples.hash_val[6] == i ? 1:0;
			valid_r[7] = tuples.hash_val[7] == i ? 1:0;


			uchar opcode = valid_r[0] + (valid_r[1] << 1) + (valid_r[2] << 2) + (valid_r[3] << 3) 
				+ (valid_r[4] << 4) + (valid_r[5] << 5) + (valid_r[6] << 6) + (valid_r[7] << 7);

			shuffled_type shuff_ifo = decoder(opcode);

			filter_type filter;
			filter.num = shuff_ifo.num;
			idx[0] = shuff_ifo.idx & 0x7;
			idx[1] = (shuff_ifo.idx >> 3) & 0x7;
			idx[2] = (shuff_ifo.idx >> 6) & 0x7;
			idx[3] = (shuff_ifo.idx >> 9) & 0x7;
			idx[4] = (shuff_ifo.idx >> 12) & 0x7;
			idx[5] = (shuff_ifo.idx >> 15) & 0x7;
			idx[6] = (shuff_ifo.idx >> 18) & 0x7;
			idx[7] = (shuff_ifo.idx >> 21) & 0x7;
			
//			if(data_r.s1 == ENDFLAG)
//				filter.end = 1;
//			else
//				filter.end = 0;
			
			uint2 data_r_uint2[8];
			
			data_r_uint2[0].x = data_r.s0;
			data_r_uint2[0].y = data_r.s1;
			data_r_uint2[1].x = data_r.s2;
			data_r_uint2[1].y = data_r.s3;
			data_r_uint2[2].x = data_r.s4;
			data_r_uint2[2].y = data_r.s5;
			data_r_uint2[3].x = data_r.s6;
			data_r_uint2[3].y = data_r.s7;
			data_r_uint2[4].x = data_r.s8;
			data_r_uint2[4].y = data_r.s9;
			data_r_uint2[5].x = data_r.sa;
			data_r_uint2[5].y = data_r.sb;
			data_r_uint2[6].x = data_r.sc;
			data_r_uint2[6].y = data_r.sd;
			data_r_uint2[7].x = data_r.se;
			data_r_uint2[7].y = data_r.sf;

			#pragma unroll 8
			for(int j = 0; j < 8; j ++){  
				uchar idx_t = idx[j];
				filter.data[j] = data_r_uint2[idx_t];
			}
			
			//if(opcode | (data_r.s1 == ENDFLAG)){
			if(opcode){
				write_channel_altera(toFilterCh[i], filter);
			}
		}
	}   
}

__attribute__((task))
__kernel void filter(){
	while(true){ 
		filter_type filter = read_channel_altera(toFilterCh[0]);
//		if(filter.end){
//			write_channel_altera(filterFlagCh[0], ENDFLAG);
//		}
//		else{
			for(int j = 0; j < filter.num; j ++){ 
				write_channel_altera(buildCh[0], filter.data[j]);
			}
//		}
	}
}

__attribute__((task))
__kernel void filter1(){
	while(true){ 
		filter_type filter = read_channel_altera(toFilterCh[1]);
//		if(filter.end){
//			write_channel_altera(filterFlagCh[1], ENDFLAG);
//		}
//		else{
			for(int j = 0; j < filter.num; j ++){ 
				write_channel_altera(buildCh[1], filter.data[j]);
			}
//		}
	}
}

__attribute__((task))
__kernel void filter2(){
	while(true){ 
		filter_type filter = read_channel_altera(toFilterCh[2]);
//		if(filter.end){
//			write_channel_altera(filterFlagCh[2], ENDFLAG);
//		}
//		else{
			for(int j = 0; j < filter.num; j ++){ 
				write_channel_altera(buildCh[2], filter.data[j]);
			}
//		}
	}
}

__attribute__((task))
__kernel void filter3(){
	while(true){ 
		filter_type filter = read_channel_altera(toFilterCh[3]);
//		if(filter.end){
//			write_channel_altera(filterFlagCh[3], ENDFLAG);
//		}
//		else{
			for(int j = 0; j < filter.num; j ++){ 
				write_channel_altera(buildCh[3], filter.data[j]);
			}
//		}
	}
}

__attribute__((task))
__kernel void filter4(){
	while(true){ 
		filter_type filter = read_channel_altera(toFilterCh[4]);
//		if(filter.end){
//			write_channel_altera(filterFlagCh[4], ENDFLAG);
//		}
//		else{
			for(int j = 0; j < filter.num; j ++){ 
				write_channel_altera(buildCh[4], filter.data[j]);
			}
//		}
	}
}

__attribute__((task))
__kernel void filter5(){
	while(true){ 
		filter_type filter = read_channel_altera(toFilterCh[5]);
//		if(filter.end){
//			write_channel_altera(filterFlagCh[5], ENDFLAG);
//		}
//		else{
			for(int j = 0; j < filter.num; j ++){ 
				write_channel_altera(buildCh[5], filter.data[j]);
			}
//		}
	}
}

__attribute__((task))
__kernel void filter6(){
	while(true){ 
		filter_type filter = read_channel_altera(toFilterCh[6]);
//		if(filter.end){
//			write_channel_altera(filterFlagCh[6], ENDFLAG);
//		}
//		else{
			for(int j = 0; j < filter.num; j ++){ 
				write_channel_altera(buildCh[6], filter.data[j]);
			}
//		}
	}
}

__attribute__((task))
__kernel void filter7(){
	while(true){ 
		filter_type filter = read_channel_altera(toFilterCh[7]);
//		if(filter.end){
//			write_channel_altera(filterFlagCh[7], ENDFLAG);
//		}
//		else{
			for(int j = 0; j < filter.num; j ++){ 
				write_channel_altera(buildCh[7], filter.data[j]);
			}
//		}
	}
}


__attribute__((task))
__kernel void hashjoin (
		__global uint * restrict matchedTable 
		//const uint cur_part
		)
{
	// build phrase 
	//__local uint relR_l [RELR_L_SIZE];
	uint2 hashtable_l [HASHTABLE_L_SIZE >> NUM_FPGA_DATAPATH_BITS][NUM_FPGA_DATAPATH];
//#if NUM_FPGA_DATAPATH == 16
//	uchar hashtable_bucket_cnt [HASHTABLE_BUCKET_NUM >> NUM_FPGA_DATAPATH_BITS][NUM_FPGA_DATAPATH];
//#endif

	bool engine_finish[NUM_FPGA_DATAPATH]; 
	//uint filterEndFlag[NUM_FPGA_DATAPATH]; 
	bool filterEndFlag[NUM_FPGA_DATAPATH]; 

	//write_channel_altera(start_channel, ENDFLAG);

	#pragma unroll NUM_FPGA_DATAPATH
	for(int j = 0; j < NUM_FPGA_DATAPATH; j ++){
		engine_finish[j] = false;
		filterEndFlag[j] = false;
	}

	while(true){
		#pragma unroll NUM_FPGA_DATAPATH
		for(int i = 0; i < NUM_FPGA_DATAPATH; i ++){ 
			// each collect engine do their work
			// low is active
			uint2 tmp_data = read_channel_nb_altera (buildCh[i], &engine_finish[i]);
			if(engine_finish[i]){
				uint key  = tmp_data.x; 
				uint val  = tmp_data.y;

				if (val == ENDFLAG) {					
					filterEndFlag[i] = 1;
				#if DEBUG
					printf("Found endflag in R. i = %d\n", i);
				#endif
				}

				uint hash_idx = HASH ((key),(HASHTABLE_BUCKET_NUM - 1),NUM_FPGA_DATAPATH_BITS);
				
				hashtable_l[hash_idx][i]= tmp_data;
			}

			//bool valid_endflag;
			//uint tmp_flag = read_channel_nb_altera (filterFlagCh[i], &valid_endflag);
			//if(valid_endflag) filterEndFlag[i] = tmp_flag;
		}

		bool all_finish = engine_finish[0] | engine_finish[1] | engine_finish[2] | engine_finish[3] | 
			engine_finish[4] | engine_finish[5] | engine_finish[6] | engine_finish[7];

		uint valid_endflag = filterEndFlag[0] & filterEndFlag[1] & filterEndFlag[2] & filterEndFlag[3] & 
			filterEndFlag[4] & filterEndFlag[5] & filterEndFlag[6] & filterEndFlag[7];

		//if(valid_endflag == ENDFLAG && !all_finish) break;
		if(valid_endflag && !all_finish) break;
	}

	#if DEBUG
		printf("Done build.\n");
	#endif

	//  =--------------------------------------probe phrase--------------------------------------------//
	uint matchedCnt[NUM_FPGA_DATAPATH][NUM_FPGA_DATAPATH];
	bool sengine_finish[NUM_FPGA_DATAPATH]; 
	//uint sfilterEndFlag[NUM_FPGA_DATAPATH]; 
	uint sfilterEndFlag = 0;

	#pragma unroll NUM_FPGA_DATAPATH
	for(int i = 0; i < NUM_FPGA_DATAPATH; i++){
		sengine_finish[i] = false;
		//sfilterEndFlag[j] = false;

		#pragma unroll NUM_FPGA_DATAPATH
		for (int j = 0; j < NUM_FPGA_DATAPATH; j++) {
			matchedCnt[i][j] = 0;
		}
	}
	
	#if DEBUG
		int while_loop_counter = 0;
	#endif

	while(1){
		//bool sengine_finish[NUM_FPGA_DATAPATH] = {false};

		#pragma unroll NUM_FPGA_DATAPATH
		for(int i = 0; i < NUM_FPGA_DATAPATH; i ++){
			uint2 s_data[8];
			bool s_valid[8];

			#pragma unroll NUM_FPGA_DATAPATH
			for (int j = 0; j < NUM_FPGA_DATAPATH; j++) {
				s_data[j] = read_channel_nb_altera(relS[j][i], &s_valid[j]);

				if (s_valid[j]) {
					uint key = s_data[j].x;
					uint hash_idx = HASH ((key),(HASHTABLE_BUCKET_NUM - 1),NUM_FPGA_DATAPATH_BITS);	
					uint hashtable_key = hashtable_l[hash_idx][i].x;

					if (key == hashtable_key) {
						matchedCnt[i][j]++;
					}
				}
			}

			sengine_finish[i] = s_valid[0] | s_valid[1] | s_valid[2] | s_valid[3] | 
								s_valid[4] | s_valid[5] | s_valid[6] | s_valid[7];
		}

		bool all_finish = sengine_finish[0] | sengine_finish[1] | sengine_finish[2] | sengine_finish[3] | 
							sengine_finish[4] | sengine_finish[5] | sengine_finish[6] | sengine_finish[7];

		bool valid_sEndflag = false;
		//uint sEndFlagData;
		uint sEndFlag = read_channel_nb_altera(s_endFlagCh, &valid_sEndflag);
		if (valid_sEndflag) sfilterEndFlag = sEndFlag;

		if (sfilterEndFlag == ENDFLAG && !all_finish) break;
		//if (sEndFlagData == ENDFLAG) break;
		#if DEBUG
			while_loop_counter++;
			//printf("Probe while loop. while loop counter = %d\n", while_loop_counter);
		#endif
	}

	uint total_cnt = 0;

	#pragma unroll NUM_FPGA_DATAPATH
	for(int i = 0; i < NUM_FPGA_DATAPATH; i ++){
		#pragma unroll NUM_FPGA_DATAPATH
		for (int j = 0; j < NUM_FPGA_DATAPATH; j++) {
			total_cnt += matchedCnt[i][j];//hashtable_l[1][i].x;
		}
	}
	matchedTable[0] += total_cnt;
}
