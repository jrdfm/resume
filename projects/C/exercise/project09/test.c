#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "split.h"
 char **remove_word(int n,char** array);
 char **rmv_word(int n,char** array);
 void free_wrd(char *name_list[]);


int main() {



 
  int i;
  char *str;
  char **wrd;

 
  char infile[25],outfile[25];
   /*
    char **rmwrd;
  str = "Goodbye cruel world!  Memory could not be allocated.";
 
 
  */
 str = "./ourtest1.public09.x < public09.input1 > public09.output1";
  wrd = split(str);
  

i = 0;

while (wrd[i] != NULL)
{
    printf("%s\n",wrd[i]);
    i++;
}
      
      i = 0;
     while (wrd[i] != NULL)
     {
       if (strcmp(wrd[i],"<") == 0)
       {
         strcpy(infile,wrd[i+1]);
         wrd = remove_word(i,wrd);
         
       }
       if ((wrd[i] != NULL) && strcmp(wrd[i],">") == 0)
       {
          strcpy(outfile,wrd[i+1]);
         wrd = remove_word(i,wrd);
         
       }

       i++;
       
     }
     

   



 
printf("%s\n",".............................");
printf("%s%s%s\n",infile," , ",outfile);

i=0;

while (wrd[i] != NULL)
{
    printf("%s\n",wrd[i]);
    i++;
}

/*
rmwrd = wrd;


rmwrd = remove_word(1,wrd);

i=0;

while (rmwrd[i] != NULL)
{
    printf("%s\n",rmwrd[i]);
    i++;
}
i = 0;
free_wrd(rmwrd);


*/
free_wrd(wrd);
    return 0;
}


char **remove_word(int n,char** array) {

char *temp,*temp2;
int i;
i = 0;
temp = array[n];
temp2 = array[n+1];
 
while (array[i+1] != NULL)
{
   
 if (i >= n)
 {


 array[i] = array[i+2];
 }
 
  
  i++; 
}

free(temp);
free(temp2);
return array;

}

void free_wrd(char *name_list[]) {

  if (name_list != NULL) {
    int i;
    i = 0;
    while (name_list[i] != NULL) {
      free(name_list[i]);
      i++;
    }
    free(name_list[i]);
    free(name_list);
  }
}