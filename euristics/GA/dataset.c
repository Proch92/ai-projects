#include "stdio.h"
#include "stdlib.h"

#define SPACE_WIDTH 1000

int main(int argc, char** argv) {
	srand(time(NULL));

	int nloc = atoi(argv[1]);

	FILE* fout = fopen(argv[2], "w");
	fwrite(&nloc, sizeof(int), 1, fout);
	int i;
	for(i=0; i!=nloc; i++) {
		double x = rand() % SPACE_WIDTH; 
		double y = rand() % SPACE_WIDTH; 
		fwrite(&x, sizeof(double), 1, fout);
		fwrite(&y, sizeof(double), 1, fout);
	}

	fclose(fout);
}
