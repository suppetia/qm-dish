subroutine set_eye(id, dim)
    !use, intrinsic :: iso_fortran_env, only: int8
    implicit none
    integer, parameter :: dp = 8

    integer, intent(in) ::  dim
    real(dp), intent(inout) :: id(dim, dim)

    integer :: i
    id = 0.0_dp
    do i = 1, dim
        id(i,i) = 1.0_dp
    end do
end subroutine set_eye

subroutine adams(y, g, coeffs_am, h, k, num_points, dim)
    !use, intrinsic :: iso_fortran_env, only: int8, int32, int64 !,dp=>real64
    implicit none

    integer, parameter :: dp = 8
    integer, parameter :: int8 = 1
    integer, parameter :: int32 = 4
    integer, parameter :: int64 = 8


    ! parameter declarations
    integer, intent(in) :: k
    integer, intent(in) :: num_points
    integer, intent(in) :: dim
    real(dp), intent(in) :: h
    integer(int64), intent(in) :: coeffs_am(k+2)
    real(dp), intent(in) :: g(num_points, dim, dim)
    real(dp), intent(inout) :: y(num_points, dim)

    ! declarations of the variables/functions used in this subroutine
    real(dp) :: M(dim, dim)
    real(dp) :: f(num_points, dim)
    integer :: i, n
    real(dp) :: scaled_summed_f(dim)
    real(dp) :: b(dim)
    integer :: ipiv(dim) ! pivot indices defining the permutation matrix P (see dgesv docs)
    integer :: info

    !print*, coeffs_am(:)

    do i = 1, k
        f(i, :) = matmul(g(i, :, :), y(i, :))
        !print*, f(i, :)
    end do

    do n = k + 1, num_points
        call set_eye(M, dim)
        M = M - h * real(coeffs_am(k+2), dp) / coeffs_am(1) * g(n,:,:)
        !print *, M
        scaled_summed_f = 0.0_dp
        do i = 1, k
            scaled_summed_f = scaled_summed_f + coeffs_am(i+1) * f(n-k+i-1, :)
        end do
        !print*, f(1:n, :)
        !print*, scaled_summed_f
        !exit
        b = y(n-1, :) + h/coeffs_am(1) * scaled_summed_f
        call dgesv(dim, 1, M, dim, ipiv, b, dim, info)
        if (info.ne.0) then
            print *, "failed to solve the linear system of equations"
        end if
        y(n, :) = b
        f(n, :) = matmul(g(n, :,:), y(n, :))
        !print*, y(n, :)

    end do

end subroutine adams