# -*- mode: text -*-

#
#


     .text
main: li      $t0,-1
      sw      $t0,ans

      li    $v0,5
      syscall
      move  $t0,$v0 # t0 = length
      sw    $t0,length 

      li    $v0,5
      syscall
      move  $t1,$v0 # t1 = width
      sw    $t1,width 

      li    $v0,5
      syscall
      move  $t2,$v0 # t2 = height
      sw    $t2,height 

if:   bltz  $t0,end # if(length < 0 ) branch to end
      bltz  $t1,end  # if(width < 0 ) branch to end
      bltz  $t2,end  # if(height < 0 ) branch to end

      mul   $t3,$t0,$t1 # $t3 = length * width
      mul   $t4,$t1,$t2 # $t4 = width * height 
      mul   $t5,$t2,$t0 # $t5 = length * height
      
      add   $t3,$t3,$t4 # $t3 = length * width + width * height 
      add   $t3,$t3,$t5 # $t3 = length * width + width * height 
                        #          + length * height
      mul   $t3,$t3,2 # ans = 2 * (width * length + length * height 
                      #       + width * height)
      sw    $t3,ans  
      
      

end:  move    $a0, $t3
      li    $v0,1 # print ans
      syscall
      li    $v0,11 # print '\n'
      li    $a0,10
      syscall

      li    $v0,10 # exit
      syscall


      
      .data

length: .word  0
width: .word   0
height: .word   0
ans: .word   0



