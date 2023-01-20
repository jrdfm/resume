#include "student.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>




int main(void) {
  

Student s = *create_student("Ryan","Lion");

printf("%s%c%s",s.first_name,' ',s.last_name);


  /*
  Student student= {NULL, NULL};

  const char last [] = "Ryan";
  const char first [] = "Lion";

printf("%d%c%d",(int)sizeof(student.first_name),' ',(int)sizeof(student.last_name));

init_student(&student, "Ryan", "Lion");

printf("%s\n%s\n",student.first_name,student.last_name);
printf("%d%c%d\n",(int)strlen(student.first_name),' ',(int)strlen(student.last_name));
printf("%d%c%d",(int)sizeof(student.first_name),' ',(int)sizeof(student.last_name));
 */
  return 0;
}





void init_student(Student *const student, const char first_name[],
                  const char last_name[]) {

                    
                      char *first = malloc(strlen(first_name)*sizeof(char));
                      char *last = malloc(strlen(last_name)*sizeof(char));
                      strcpy(first,first_name);
                      strcpy(last,last_name);

                    if (student != NULL)
                    {
                      student ->first_name = first;
                      student ->last_name = last;

                    }
                    
                     
                      
                  }




void print_student(Student student) {

if (student.last_name != NULL)
{
    printf("%s%s",student.last_name,", ");
}

if (student.first_name != NULL)
{
    printf("%s",student.first_name);
}


}

Student *create_student(const char first_name[], const char last_name[]) {

Student *new = malloc(sizeof(Student));
char *first = malloc(strlen(first_name)*sizeof(char));
char *last = malloc(strlen(last_name)*sizeof(char));
                      strcpy(first,first_name);
                      strcpy(last,last_name);

                      new ->first_name = first;
                      new->last_name = last;

                

return new;

}



unsigned int change_first_name(Student *const student, const char new_name[]) {

 
                     
                    if (student != NULL && new_name != NULL)
                    {
                     char *first = malloc(strlen(new_name)*sizeof(char));
                      
                      strcpy(first,new_name);
                      free(student->first_name);
                      student ->first_name = first;
                     
                      return 1;
                    }

                    return 0;
                    


}
unsigned int change_last_name(Student *const student, const char new_name[]) {

if (student != NULL && new_name != NULL)
                    {
                     char *last = malloc(strlen(new_name)*sizeof(char));
                      
                      strcpy(last,new_name);
                      free(student->last_name);
                      student ->last_name = last;
                     
                      return 1;
                    }

                    return 0;



}


          
              /*    



unsigned int add_nickname(Student *const student, const char nickname[]) {}
              
              
int compare(Student student1, Student student2) {



}  
unsigned int reset_student(Student *const student) {} 
unsigned int free_student(Student **const student) {} 




*/