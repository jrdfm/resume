#include <stdio.h>
#include <stdlib.h>

#define SIZE 512
int main(int argc, char *argv[]) {
    FILE *fs,*fd;
    int n;
    char buf[SIZE];

    if (argc != 3)
    {
        exit(-1);
    }
    fs = fopen(argv[1],"rb");
    fd = fopen(argv[2],"wb");

    if (fs == NULL || fd == NULL)
    {
        exit(-1);
    }

    while ((n = fread(buf,1,SIZE,fs)) > 0)
    {
        fwrite(buf,1,n,fd);
    }
    
    fclose(fs);
    fclose(fd);

    return 0;
    

}