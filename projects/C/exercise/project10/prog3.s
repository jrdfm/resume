# -*- mode: text -*-

#
#

     .data 

number: .word 0
result: .word 0


     .text

main: li    $sp, 0x7ffffffc  # set up stack pointer
      li    $v0,5 # read in number
      syscall
      move  $t0,$v0
      sw    $t0,number
      sw    $t0,($sp) # push number onto stack
      sub   $sp,$sp,4

      jal   jacobsthal  # call jacobsthal

      add   $sp,$sp,4  # pop number 
      move  $t0,$v0    # store return to result
      sw    $t0,result 

      move  $a0, $t0 # print result
      li    $v0,1 
      syscall
      
      li    $v0,11 # print '\n'
      li    $a0,10
      syscall

      li    $v0,10
      syscall

                                    
                                     # prologue
jacobsthal:     sub     $sp, $sp, 20      # set new stack pointer
                sw      $ra, 20($sp)      # save return address in stack
                sw      $fp, 16($sp)      # save old frame pointer in stack
                add     $fp, $sp, 20      # set new frame pointer

                li      $t0,-1
                sw      $t0,12($sp)    # ans = -1

                lw      $t1,4($fp)    # get n in caller's frame
                

if:         bltz     $t1,endif  # if(n < 0 ) end

            seq     $t2,$t1,0 # $t2 = (n == 0)
            seq     $t3,$t1,1 # $t3 = (n == 1)
            or      $t2,$t2,$t3 # t2 = (n == 0 || n == 1)

            beqz    $t2,else 

            
            sw      $t1,12($sp)    # ans = n
        
            j       endif

else:       lw      $t1,4($fp) # get n from callers frame
            sub     $t4, $t1, 2      # push arg. n - 2 onto stack
            sw      $t4, ($sp)
            sub     $sp, $sp, 4

            jal     jacobsthal 
            add     $sp, $sp, 4      # pop arg. from stack

            move    $t0,$v0 # temp1 =  jacobsthal(n-2)
            mul     $t0,$t0,2
            sw      $t0,8($sp) 

            lw      $t1,4($fp) # get n from callers frame
            sub     $t3, $t1, 1      # push arg. n - 1 onto stack
            sw      $t3, ($sp)
            sub     $sp, $sp, 4

            jal     jacobsthal 
            add     $sp, $sp, 4      # pop arg. from stack

            move    $t3,$v0 # temp2 =  jacobsthal(n-1)
            sw      $t3,4($sp)

            lw      $t0,8($sp)
            lw      $t3,4($sp)
            

            add     $t6,$t3,$t0  # ans = temp1 + temp2
            sw      $t6,12($sp)

            j       endif

    

endif:    
            lw      $t7,12($sp)
            move    $v0, $t7                 
                                     # epilogue
            lw      $ra, 20($sp)      # load return address from stack
            lw      $fp, 16($sp)      # restore old frame pointer from stack
            add     $sp, $sp, 20      # reset stack pointer
            jr      $ra              # return to caller using saved return address