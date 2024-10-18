module FModule
    implicit none

    real(8), parameter :: a = 1.0
    real(8), parameter :: b = 100.0
    real(8), parameter :: epsilon = 0.000001
    real(8), parameter :: s = 12.0

  contains
     elemental function f(x) result(y)
 
         implicit none
         real(8), intent(in) :: x
         real(8) ::  y, temp
         integer :: i, j
         
         y = 0.0
         do i = 100, 1, -1
             temp = 0.0
             do j = i, 1, -1
                 temp = temp + (x + 0.5 * j)**(-3.3)
             end do
             y = y + DSIN(x + temp) / (1.3**i)
         end do
     
     end function f
 end module FModule