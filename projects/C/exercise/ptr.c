# include<stdio.h>
#include <stdlib.h>
#include <string.h>


int main() {

    char *cptr;

    char str[15]="Yared is Great";

    void *ptr;

    ptr = &str;

    cptr = malloc(strlen((char*)ptr) + 1);
    strcpy(cptr,((char*)ptr));


    printf("%s\n",cptr);






    return 0;
}

