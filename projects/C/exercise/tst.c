#include <stdio.h>
int main() {
    int i = 0;
    i = 1000;
   int count = 15;

    for (int j = 0; j < count; j++)
    {
        i =i-10*(j);
    }
    
   printf("%d\n",i);
   printf("Hello, World!");
   return 0;
}