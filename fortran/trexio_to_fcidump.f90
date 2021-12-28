program trexio_to_fcidump
  use trexio
  implicit none

  ! This program reads integrals from a trexio file and writes a FCIDUMP file
  ! It needs to get the number of active MOs from the command-line:
  !
  ! trexio_to_fcidump TREXIO_FILE istart iend
  !
  ! Where istart and iend are the first and last active MOs to consider.
  ! If these arguments are absent, the default is to consider all MOs as active

  character(64)    :: argument
  character(256)   :: trexio_filename
  integer          :: i,j,k,l
  integer          :: ii,jj,kk,ll
  integer          :: istart, iend, iunit
  double precision :: e0
  character(2)    , allocatable :: A(:)
  double precision, allocatable :: h0(:,:), int2(:,:,:), int3(:,:,:)

  integer(8) :: trexio_file
  integer    :: rc
  integer    :: up_num, dn_num, elec_num
  integer    :: mo_num, n_act

  integer(8), parameter    :: BUFSIZE = 100000_8
  integer(8)               :: offset, icount, size_max, m
  integer                  :: buffer_index(4,BUFSIZE)
  double precision         :: buffer_values(BUFSIZE)

  if(command_argument_count() < 1) then
    print *, 'syntax: trexio_to_fcidump TREXIO_FILE [ istart iend ]'
    call exit(-1)
  end if

  call get_command_argument(1, trexio_filename)
  trexio_file = trexio_open(trim(trexio_filename), 'r', TREXIO_TEXT, rc)
  if (rc == TREXIO_OPEN_ERROR) then
    trexio_file = trexio_open(trim(trexio_filename), 'r', TREXIO_HDF5, rc)
  end if
  call check_error(rc)

  rc = trexio_read_mo_num(trexio_file, mo_num)
  call check_error(rc)

  if(command_argument_count() == 3) then
     call get_command_argument(2, argument)
     read(argument, *) istart
     call get_command_argument(3, argument)
     read(argument, *) iend
  else
     istart = 1
     iend   = mo_num
  end if

  if (istart < 1) then
     print *, 'Error: istart should be > 0'
     call exit(-1)
  end if

  if (iend < 1) then
     print *, 'Error: iend should be > 0'
     call exit(-1)
  end if

  if (istart > mo_num) then
     print *, 'Error: istart should be < ', mo_num
     call exit(-1)
  end if

  if (iend > mo_num) then
     print *, 'Error: iend should be < ', mo_num
     call exit(-1)
  end if

  if (istart >= iend) then
     print *, 'Error: istart should be < iend'
     call exit(-1)
  end if

  n_act = iend - istart + 1

  rc = trexio_read_electron_up_num(trexio_file, up_num)
  call check_error(rc)

  rc = trexio_read_electron_dn_num(trexio_file, dn_num)
  call check_error(rc)

  elec_num = up_num + dn_num

  open(newunit=iunit, file=trim(trexio_filename)//'.FCIDUMP', &
       FORM='FORMATTED')

  write(iunit,*) &
       '&FCI NORB=', n_act, ', NELEC=', elec_num-(istart-1)*2, &
       ', MS2=', (up_num-dn_num), ','

  ! TODO: Implement symmetries here
  allocate(A(n_act))
  A(1:n_act) = '1,'
  write(iunit,*) 'ORBSYM=', (A(i), i=1,n_act)
  write(iunit,*) 'ISYM=0,'
  write(iunit,*) '&end'
  deallocate(A)

  rc = trexio_has_mo_1e_int_core_hamiltonian(trexio_file)
  if (rc /= TREXIO_SUCCESS) then
     print *, 'No core hamiltonian in file'
     call check_error(rc)
  end if

  allocate(h0(mo_num,mo_num), int3(n_act,n_act,2), int2(istart,istart,2))
  int3 = 0.d0
  int2 = 0.d0

  rc = trexio_has_mo_2e_int_eri(trexio_file)
  if (rc /= TREXIO_SUCCESS) then
     print *, 'No electron repulsion integrals in file'
     call check_error(rc)
  end if

  rc = trexio_read_mo_2e_int_eri_size (trexio_file, size_max)
  call check_error(rc)

  offset = 0_8
  icount = BUFSIZE
  do while (icount == BUFSIZE)
    if (offset < size_max) then
      rc = trexio_read_mo_2e_int_eri(trexio_file, offset, icount, buffer_index, buffer_values)
      offset = offset + icount
    else
      icount = 0
    end if
    do m=1,icount
      ii = buffer_index(1,m)
      jj = buffer_index(2,m)
      kk = buffer_index(3,m)
      ll = buffer_index(4,m)

      i = ii - istart + 1
      j = jj - istart + 1
      k = kk - istart + 1
      l = ll - istart + 1

      if (i > 0 .and. i <= n_act .and. &
          j > 0 .and. j <= n_act .and. &
          k > 0 .and. k <= n_act .and. &
          l > 0 .and. l <= n_act) then
         write(iunit, '(E24.15, X, 4(I6, X))') buffer_values(m), i, j, k, l
      end if

      if (i > 0 .and. i <= n_act .and. &
          k > 0 .and. k <= n_act .and. &
          jj < istart .and. jj == ll) then
         int3(i,k,1) = buffer_values(m)
         int3(k,i,1) = int3(i,k,1)
      end if

      if (i > 0 .and. i <= n_act .and. &
          l > 0 .and. l <= n_act .and. &
          jj < istart .and. jj == kk) then
         int3(i,l,2) = buffer_values(m)
         int3(l,i,2) = int3(i,l,2)
      else if (i > 0 .and. i <= n_act .and. &
               j > 0 .and. j <= n_act .and. &
               ll < istart .and. ll == kk) then
         int3(i,j,2) = buffer_values(m)
         int3(j,i,2) = int3(i,j,2)
      end if

      if (j > 0 .and. j <= n_act .and. &
          l > 0 .and. l <= n_act .and. &
          ii < istart .and. ii == kk) then
         int3(j,l,1) = buffer_values(m)
         int3(l,j,1) = int3(j,l,1)
      end if

      if (j > 0 .and. j <= n_act .and. &
          k > 0 .and. k <= n_act .and. &
          ii < istart .and. ii == ll) then
         int3(j,k,2) = buffer_values(m)
         int3(k,j,2) = int3(j,k,2)
      else if (l > 0 .and. l <= n_act .and. &
               k > 0 .and. k <= n_act .and. &
               ii < istart .and. ii == jj) then
         int3(l,k,2) = buffer_values(m)
         int3(k,l,2) = int3(l,k,2)
      end if

      if (ii < istart .and. ii == kk .and. &
          jj < istart .and. jj == ll) then
         int2(ii,jj,1) = buffer_values(m)
         int2(jj,ii,1) = int2(ii,jj,1)
      end if

      if (ii < istart .and. ii == ll .and. &
          jj < istart .and. jj == kk) then
         int2(ii,jj,2) = buffer_values(m)
         int2(jj,ii,2) = int2(ii,jj,2)
      else if (ii < istart .and. ii == jj .and. &
               kk < istart .and. kk == ll) then
         int2(ii,kk,2) = buffer_values(m)
         int2(kk,ii,2) = int2(ii,kk,2)
      end if

    end do

  end do

  h0   = 0.d0

  rc = trexio_read_mo_1e_int_core_hamiltonian(trexio_file, h0)
  call check_error(rc)

  ! Add core Fock operator
  do j=1,n_act
     jj = j + istart - 1
     do i=1,n_act
        ii = i + istart - 1
        int3(i,j,1) = h0(ii,jj) + 2.d0*int3(i,j,1) - int3(i,j,2)
     end do
  end do

  ! One-e integrals
  do j=1,n_act
     do i=j,n_act
        if (int3(i,j,1) /= 0.d0) then
           write(iunit, '(E24.15, X, 4(I6, X))') int3(i,j,1), i, j, 0, 0
        end if
     end do
  end do

  e0 = 0.d0
  rc = trexio_read_nucleus_repulsion(trexio_file, e0)
  call check_error(rc)
  do i=1,istart-1
     e0 = e0 + 2.d0*h0(i,i)
     do j=1,istart-1
        e0 = e0 + 2.d0*int2(i,j,1) - int2(i,j,2)
     end do
  end do
  deallocate(int2,int3)

  write(iunit, '(E24.15, X, 4(I6, X))') e0, 0, 0, 0, 0

  close(iunit)

end program trexio_to_fcidump


subroutine check_error(rc)
  use trexio
  implicit none
  integer, intent(in) :: rc
  character(128)      :: message
  if (rc /= TREXIO_SUCCESS) then
     call trexio_string_of_error(rc, message)
     print *, 'TREXIO error: '
     print *, trim(message)
     stop rc
  end if
end subroutine check_error
