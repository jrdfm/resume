/*
This program checks for lines that are more than 
80 characters.If found prints an * followed by
line number and the line. If the line is less
than 80 lines just the line number with its
content.
 */
#include<stdio.h>

int main(void) {

   char c[99999];
   char temp;
   int i,j,count,line,k;  
   
   i=0;
    
   while (scanf("%c", &temp) != EOF) {
         c[i] = temp;
         i++;
      }	
      	 count = 0;
         line = 0;
        
    for (j = 0;j < i;j++) {
      char ch[99999];
      if (c[j] != '\n')
      { 
        if (c[j] == '\t') 
	  count+=7;
             
        ch[j]=c[j];
        count++;
      }
        else { 
          line++;
        if (count != 0 && count <= 80) {
           printf("%c", ' ');       
           printf("%5d", line);
           printf(": ");
          for (k = j-count;k < j;k++) {
            printf("%c", ch[k]);
             } 
      	    printf("\n");         
            count = 0;
             }
         else if (count > 80) {
	    printf("*");
            printf("%5d", line);
            printf(": ");
	   for(k=j-count;k<j;k++) {
            printf("%c",ch[k]);
            } 
            printf("\n");
	    printf("%88c", ' ');
	   for (k = 0;k < (count+8)-88;k++) {
	    printf("%c", '^');
             }
            printf("\n");    
            count = 0;
             }
	    count = 0; 
               
             } 
          } 

return 0;
}
