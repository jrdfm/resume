/*
This Program reads input from warnlines
and prints only the line numbers of 
length above 80.

 */

#include<stdio.h>
int main(void) {

  
  char temp;
  int d;

  while (scanf("%c", &temp) != EOF) {

      if (temp != ' ')
	{
      if (temp == '*')
	{
	  scanf("%d", &d);
          printf("%d\n", d); 	

        }
        } 
      
     }
     printf("\n");
      
     return 0;
}
