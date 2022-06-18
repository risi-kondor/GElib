import gelib

v=gelib.SO3vec.Frandn(1,6) # Test that up to a constant FFT and iFFT are inverses of each other
f=gelib.SO3iFFT(v,20)
w=gelib.SO3FFT(f,6)

print(w.parts[2]/v.parts[2])
