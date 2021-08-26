#include <stdio.h>
#include <trexio.h>


int main(int argc, char** argv) {
    
    char* filename = argv[1];
    printf("File name: %s\n", filename);

    trexio_t* trexio_file = trexio_open(filename, 'r', TREXIO_TEXT);
    
    int n;
    trexio_exit_code rc = trexio_read_nucleus_num(trexio_file, &n);
    printf("Num: %d\n",n);

}

