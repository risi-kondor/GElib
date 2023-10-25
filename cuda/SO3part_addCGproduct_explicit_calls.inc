  switch(l1){
      case 0:
        switch(l2){
        case 0:
          switch(l){
          case 0: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_0_0_0><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          }
        break;
        case 1:
          switch(l){
          case 1: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_0_1_1><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          }
        break;
        case 2:
          switch(l){
          case 2: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_0_2_2><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          }
        break;
        }

      break;
      case 1:
        switch(l2){
        case 0:
          switch(l){
          case 1: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_1_0_1><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          }
        break;
        case 1:
          switch(l){
          case 0: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_1_1_0><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          case 1: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_1_1_1><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          case 2: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_1_1_2><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          }
        break;
        case 2:
          switch(l){
          case 1: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_1_2_1><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          case 2: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_1_2_2><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          case 3: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_1_2_3><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          }
        break;
        }

      break;
      case 2:
        switch(l2){
        case 0:
          switch(l){
          case 2: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_2_0_2><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          }
        break;
        case 1:
          switch(l){
          case 1: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_2_1_1><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          case 2: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_2_1_2><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          case 3: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_2_1_3><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          }
        break;
        case 2:
          switch(l){
          case 0: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_2_2_0><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          case 1: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_2_2_1><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          case 2: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_2_2_2><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          case 3: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_2_2_3><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          case 4: SO3part_addCGproduct_explicit<SO3part_addCGproduct_explicit_kernel_2_2_4><<<b,cnine::roundup(y.n2,32),nlines*128,stream>>>(r,x,y); break;
          }
        break;
        }

      break;
      }
