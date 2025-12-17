import pytest
from types import SimpleNamespace
from petpal.preproc.register import register_pet

import petpal.preproc.register as reg_mod

class DummyImage:
    def __init__(self, dimension, name=None):
        self.dimension = dimension
        self.name = name

def make_mocks(monkeypatch, pet_dim, recorded):
    # Mock determine_motion_target to return a path string
    monkeypatch.setattr(reg_mod, "determine_motion_target", lambda motion_target_option, input_image_path: "motion_target.nii")
    # Mock ants.image_read to return DummyImage instances depending on path
    def mock_image_read(path):
        if path == "motion_target.nii":
            img = DummyImage(dimension=3, name="motion_target")
        elif path == "mri.nii":
            img = DummyImage(dimension=3, name="mri")
        elif path == "pet.nii":
            img = DummyImage(dimension=pet_dim, name="pet")
        else:
            img = DummyImage(dimension=0, name=path)
        recorded['image_reads'].append((path, img))
        return img
    monkeypatch.setattr(reg_mod.ants, "image_read", mock_image_read)
    # Mock ants.registration to record args and return transforms
    def mock_registration(*, moving, fixed, type_of_transform, write_composite_transform=True, **kwargs):
        recorded['registration_calls'].append({
            'moving': moving, 'fixed': fixed, 'type_of_transform': type_of_transform, 'kwargs': kwargs
        })
        return {'fwdtransforms': ['/fake/xfm']}
    monkeypatch.setattr(reg_mod.ants, "registration", mock_registration)
    # Mock ants.apply_transforms to record imagetype and return a dummy transformed image
    def mock_apply_transforms(*, moving, fixed, transformlist, interpolator=None, imagetype=None, **kwargs):
        recorded['apply_calls'].append({'moving': moving, 'fixed': fixed, 'transformlist': transformlist, 'imagetype': imagetype})
        return SimpleNamespace(name="transformed_image", imagetype=imagetype)
    monkeypatch.setattr(reg_mod.ants, "apply_transforms", mock_apply_transforms)
    # Mock ants.image_write to record output path
    monkeypatch.setattr(reg_mod.ants, "image_write", lambda img, out: recorded['written'].append((img, out)))
    # Mock image_io.safe_copy_meta
    monkeypatch.setattr(reg_mod.image_io, "safe_copy_meta", lambda input_image_path, out_image_path: recorded['meta_copied'].append((input_image_path, out_image_path)))

@pytest.mark.parametrize("pet_dim, expected_imagetype", [
    (4, 3),
    (3, 0),
])
def test_register_pet_sets_imagetype_and_writes(monkeypatch, capsys, pet_dim, expected_imagetype):
    recorded = {'image_reads': [], 'registration_calls': [], 'apply_calls': [], 'written': [], 'meta_copied': []}
    make_mocks(monkeypatch, pet_dim=pet_dim, recorded=recorded)

    # Call function under test
    register_pet(input_reg_image_path="pet.nii",
                 out_image_path="out.nii",
                 reference_image_path="mri.nii",
                 motion_target_option="some_option",
                 verbose=True,
                 type_of_transform="DenseRigid")

    # Check registration was called with motion target as moving and mri as fixed
    assert recorded['registration_calls'], "ants.registration was not called"
    reg_call = recorded['registration_calls'][-1]
    assert getattr(reg_call['moving'], "name", None) == "motion_target"
    assert getattr(reg_call['fixed'], "name", None) == "mri"
    assert reg_call['type_of_transform'] == "DenseRigid"

    # Check apply_transforms was called with correct imagetype based on input dimension
    assert recorded['apply_calls'], "ants.apply_transforms was not called"
    apply_call = recorded['apply_calls'][-1]
    assert apply_call['imagetype'] == expected_imagetype

    # Check image was written and metadata copied
    assert recorded['written'] and recorded['written'][-1][1] == "out.nii"
    assert recorded['meta_copied'] and recorded['meta_copied'][-1] == ("pet.nii", "out.nii")

    # Check verbose prints
    captured = capsys.readouterr().out
    print(captured)
    assert "Registration computed transforming image motion_target.nii to mri.nii space" in captured
    assert "Registration applied to pet.nii" in captured
    assert "Transformed image saved to out.nii" in captured