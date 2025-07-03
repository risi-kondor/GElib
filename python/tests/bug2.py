import torch
import gelib
import math

def get_wigner_d_for_l1_rz(angle_rad: float) -> torch.Tensor:
    """
    Computes the Wigner-D matrix for L=1 for a rotation around the Z-axis.
    This is a standard representation of the SO(3) group action on L=1 spherical tensors.
    The matrix is for m'=-1,0,1 (rows) and m=-1,0,1 (columns).
    """
    return torch.tensor([
        [math.cos(angle_rad) + 0j, -math.sin(angle_rad) + 0j, 0 + 0j],
        [math.sin(angle_rad) + 0j,  math.cos(angle_rad) + 0j, 0 + 0j],
        [0 + 0j,                    0 + 0j,                   1 + 0j]
    ], dtype=torch.cfloat)


def check_gelib_consistency():
    """
    Runs a series of checks on the GElib library to test its internal
    self-consistency for SO(3) operations.
    """
    # --- Setup ---
    # Vector pointing along the x-axis
    cartesian_vector_x = torch.tensor([[[1.0], [0.0], [0.0]]], dtype=torch.float32)
    # Vector pointing along the y-axis (result of rotating x by +90 deg around z)
    cartesian_vector_y = torch.tensor([[[0.0], [1.0], [0.0]]], dtype=torch.float32)

    try:
        # --- Pre-computation ---
        sh_x_from_gelib = gelib.SO3partArr.spharm(1, cartesian_vector_x)
        coeffs_x_from_gelib = sh_x_from_gelib.as_subclass(torch.Tensor)[0, :, :].clone()

        sh_y_from_gelib = gelib.SO3partArr.spharm(1, cartesian_vector_y)
        coeffs_y_from_gelib = sh_y_from_gelib.as_subclass(torch.Tensor)[0, :, :].clone()
        
        rz_angle = math.pi / 2
        rz_matrix = torch.tensor([
            [math.cos(rz_angle), -math.sin(rz_angle), 0.0],
            [math.sin(rz_angle),  math.cos(rz_angle), 0.0],
            [0.0,                 0.0,                1.0]
        ], dtype=torch.float32)
        rz_so3 = gelib.SO3element(rz_matrix)

    except Exception as e:
        print("FATAL ERROR: Could not execute a required GElib function during setup. Cannot proceed.")
        print(f"Error: {e}")
        return

    # --- Test 1: Identity Rotation ---
    # Applying an identity rotation must not change the tensor.
    identity_rotation = gelib.SO3element.identity()
    sh_x_rotated_by_identity = sh_x_from_gelib.apply(identity_rotation)
    coeffs_x_rotated_by_identity = sh_x_rotated_by_identity.as_subclass(torch.Tensor)[0, :, :].clone()
    identity_preserves_coeffs = torch.allclose(coeffs_x_from_gelib, coeffs_x_rotated_by_identity, atol=1e-6)
    print(sh_x_from_gelib)
    #print(coeffs_x_from_gelib)
    #print(sh_x_rotated_by_identity)
    #print(coeffs_x_rotated_by_identity)

    # --- Test 2: GElib Rotation Self-Consistency ---
    # `gelib.apply(rot, spharm(v))` should equal `spharm(rot @ v)`.
    sh_x_rotated_by_rz = sh_x_from_gelib.apply(rz_so3)
    coeffs_path_A = sh_x_rotated_by_rz.as_subclass(torch.Tensor)[0, :, :].clone()
    print(sh_x_rotated_by_rz)
    print(sh_y_from_gelib)
    rotation_is_self_consistent = torch.allclose(coeffs_path_A, coeffs_y_from_gelib, atol=1e-6)

    # --- Test 3: Manual Rotation Self-Consistency ---
    # `D(rot) @ spharm(v)` should equal `spharm(rot @ v)`.
    # This tests if `spharm` output transforms correctly under the standard Wigner-D matrix.
    wigner_d_rz = get_wigner_d_for_l1_rz(rz_angle)
    coeffs_x_manual_rotated = wigner_d_rz @ coeffs_x_from_gelib
    manual_rotation_is_self_consistent = torch.allclose(coeffs_x_manual_rotated, coeffs_y_from_gelib, atol=1e-6)


    # --- Final Summary ---
    print("\n--- GElib Self-Consistency Report ---")
    print("-" * 35)

    if not identity_preserves_coeffs:
        print("CRITICAL BUG FOUND: Identity rotation test FAILED.")
        print("   - Finding: `SO3partArr.apply(identity)` alters the input tensor's coefficients.")
    else:
        print("Identity rotation test PASSED.")
        print("   - Finding: `SO3partArr.apply(identity)` correctly preserves the input tensor's coefficients.")

    if not rotation_is_self_consistent:
        print("\n CRITICAL BUG FOUND: GElib rotation self-consistency test FAILED.")
        print("   - Finding: `gelib.apply(rot, spharm(v))` is NOT equal to `spharm(rot @ v)`.")
    else:
        print("\n GElib rotation self-consistency test PASSED.")
        print("   - Finding: The library's `apply` and `spharm` methods appear internally consistent.")

    if not manual_rotation_is_self_consistent:
        print("\n CRITICAL BUG FOUND: Manual rotation self-consistency test FAILED.")
        print("   - Finding: `D_matrix @ spharm(v)` is NOT equal to `spharm(rot @ v)`.")
    else:
        print("\n Manual rotation self-consistency test PASSED.")
        print("   - Finding: The output of GElib's `spharm` transforms as expected under a standard Wigner-D matrix.")

if __name__ == "__main__":
    check_gelib_consistency()
