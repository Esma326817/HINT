import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


# =============================================================================
# Tests for DeltaTCPPoseActions and AbsoluteTCPPoseActions
# =============================================================================


def _make_random_axis_angle():
    """Generate a random axis-angle rotation."""
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = np.random.uniform(0, np.pi)
    return axis * angle


def _make_random_quaternion():
    """Generate a random unit quaternion (w, x, y, z)."""
    q = np.random.randn(4)
    return q / np.linalg.norm(q)


def _make_random_rotation_6d():
    """Generate a random 6D rotation representation."""
    # Generate a random rotation matrix and extract first two columns
    aa = _make_random_axis_angle()
    R = _transforms.axis_angle_to_matrix(aa)
    return np.concatenate([R[:, 0], R[:, 1]])


def _make_random_euler_xyz():
    """Generate random Euler XYZ angles (radians) from a random rotation matrix."""
    aa = _make_random_axis_angle()
    R = _transforms.axis_angle_to_matrix(aa)
    return _transforms.matrix_to_euler_xyz(R)


def test_delta_tcp_pose_actions_axis_angle():
    """Test DeltaTCPPoseActions with axis-angle representation."""
    np.random.seed(42)

    # Create state: [x, y, z, ax, ay, az, gripper]
    state_pos = np.array([1.0, 2.0, 3.0])
    state_rot = _make_random_axis_angle()
    state = np.concatenate([state_pos, state_rot, [0.5]])

    # Create actions: [x, y, z, ax, ay, az, gripper] x 2 timesteps
    action1_pos = np.array([1.5, 2.5, 3.5])
    action1_rot = _make_random_axis_angle()
    action2_pos = np.array([2.0, 3.0, 4.0])
    action2_rot = _make_random_axis_angle()

    actions = np.stack([
        np.concatenate([action1_pos, action1_rot, [0.8]]),
        np.concatenate([action2_pos, action2_rot, [1.0]]),
    ])

    item = {"state": state.copy(), "actions": actions.copy()}

    # Apply delta transform
    delta_transform = _transforms.DeltaTCPPoseActions(pos_dims=3, rotation_repr="axis_angle")
    delta_item = delta_transform(item)

    # Apply absolute transform to recover original
    absolute_transform = _transforms.AbsoluteTCPPoseActions(pos_dims=3, rotation_repr="axis_angle")
    recovered_item = absolute_transform({"state": state.copy(), "actions": delta_item["actions"].copy()})

    # Check that we recover the original actions (within numerical tolerance)
    assert np.allclose(recovered_item["actions"], actions, atol=1e-5)


def test_delta_tcp_pose_actions_quaternion():
    """Test DeltaTCPPoseActions with quaternion representation."""
    np.random.seed(42)

    # Create state: [x, y, z, qw, qx, qy, qz, gripper]
    state_pos = np.array([1.0, 2.0, 3.0])
    state_rot = _make_random_quaternion()
    state = np.concatenate([state_pos, state_rot, [0.5]])

    # Create actions
    action1_pos = np.array([1.5, 2.5, 3.5])
    action1_rot = _make_random_quaternion()
    action2_pos = np.array([2.0, 3.0, 4.0])
    action2_rot = _make_random_quaternion()

    actions = np.stack([
        np.concatenate([action1_pos, action1_rot, [0.8]]),
        np.concatenate([action2_pos, action2_rot, [1.0]]),
    ])

    item = {"state": state.copy(), "actions": actions.copy()}

    # Apply delta transform
    delta_transform = _transforms.DeltaTCPPoseActions(pos_dims=3, rotation_repr="quaternion")
    delta_item = delta_transform(item)

    # Apply absolute transform to recover original
    absolute_transform = _transforms.AbsoluteTCPPoseActions(pos_dims=3, rotation_repr="quaternion")
    recovered_item = absolute_transform({"state": state.copy(), "actions": delta_item["actions"].copy()})

    # Check rotation matrices are equivalent (quaternions can differ by sign)
    for i in range(actions.shape[0]):
        orig_R = _transforms.quaternion_to_matrix(actions[i, 3:7])
        recovered_R = _transforms.quaternion_to_matrix(recovered_item["actions"][i, 3:7])
        assert np.allclose(orig_R, recovered_R, atol=1e-5)

    # Check position and gripper
    assert np.allclose(recovered_item["actions"][:, :3], actions[:, :3], atol=1e-5)
    assert np.allclose(recovered_item["actions"][:, 7:], actions[:, 7:], atol=1e-5)


def test_delta_tcp_pose_actions_rotation_6d():
    """Test DeltaTCPPoseActions with rotation_6d representation."""
    np.random.seed(42)

    # Create state: [x, y, z, r1-r6, gripper]
    state_pos = np.array([1.0, 2.0, 3.0])
    state_rot = _make_random_rotation_6d()
    state = np.concatenate([state_pos, state_rot, [0.5]])

    # Create actions
    action1_pos = np.array([1.5, 2.5, 3.5])
    action1_rot = _make_random_rotation_6d()
    action2_pos = np.array([2.0, 3.0, 4.0])
    action2_rot = _make_random_rotation_6d()

    actions = np.stack([
        np.concatenate([action1_pos, action1_rot, [0.8]]),
        np.concatenate([action2_pos, action2_rot, [1.0]]),
    ])

    item = {"state": state.copy(), "actions": actions.copy()}

    # Apply delta transform
    delta_transform = _transforms.DeltaTCPPoseActions(pos_dims=3, rotation_repr="rotation_6d")
    delta_item = delta_transform(item)

    # Apply absolute transform to recover original
    absolute_transform = _transforms.AbsoluteTCPPoseActions(pos_dims=3, rotation_repr="rotation_6d")
    recovered_item = absolute_transform({"state": state.copy(), "actions": delta_item["actions"].copy()})

    # Check rotation matrices are equivalent
    for i in range(actions.shape[0]):
        orig_R = _transforms.rotation_6d_to_matrix(actions[i, 3:9])
        recovered_R = _transforms.rotation_6d_to_matrix(recovered_item["actions"][i, 3:9])
        assert np.allclose(orig_R, recovered_R, atol=1e-5)

    # Check position and gripper
    assert np.allclose(recovered_item["actions"][:, :3], actions[:, :3], atol=1e-5)
    assert np.allclose(recovered_item["actions"][:, 9:], actions[:, 9:], atol=1e-5)


def test_delta_tcp_pose_actions_euler_xyz():
    """Test DeltaTCPPoseActions with extrinsic XYZ Euler angles (radians)."""
    np.random.seed(42)

    state_pos = np.array([1.0, 2.0, 3.0])
    state_rot = _make_random_euler_xyz()
    state = np.concatenate([state_pos, state_rot, [0.5]])

    action1_pos = np.array([1.5, 2.5, 3.5])
    action1_rot = _make_random_euler_xyz()
    action2_pos = np.array([2.0, 3.0, 4.0])
    action2_rot = _make_random_euler_xyz()

    actions = np.stack([
        np.concatenate([action1_pos, action1_rot, [0.8]]),
        np.concatenate([action2_pos, action2_rot, [1.0]]),
    ])

    item = {"state": state.copy(), "actions": actions.copy()}

    delta_transform = _transforms.DeltaTCPPoseActions(pos_dims=3, rotation_repr="euler_xyz")
    delta_item = delta_transform(item)

    absolute_transform = _transforms.AbsoluteTCPPoseActions(pos_dims=3, rotation_repr="euler_xyz")
    recovered_item = absolute_transform({"state": state.copy(), "actions": delta_item["actions"].copy()})

    for i in range(actions.shape[0]):
        orig_R = _transforms.euler_xyz_to_matrix(actions[i, 3:6])
        recovered_R = _transforms.euler_xyz_to_matrix(recovered_item["actions"][i, 3:6])
        assert np.allclose(orig_R, recovered_R, atol=1e-5)

    assert np.allclose(recovered_item["actions"][:, :3], actions[:, :3], atol=1e-5)
    assert np.allclose(recovered_item["actions"][:, 6:], actions[:, 6:], atol=1e-5)


def test_rotation_repr_euler_alias():
    """``rotation_repr='euler'`` is normalized to ``euler_xyz``."""
    t = _transforms.DeltaTCPPoseActions(rotation_repr="euler")
    assert t.rotation_repr == "euler_xyz"


def test_delta_tcp_pose_actions_noop():
    """Test that DeltaTCPPoseActions is no-op when disabled or no actions."""
    state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.5])
    actions = np.array([[1.5, 2.5, 3.5, 0.2, 0.3, 0.4, 0.8]])

    item = {"state": state, "actions": actions}

    # No-op when disabled
    transform = _transforms.DeltaTCPPoseActions(enabled=False)
    assert transform(item) is item

    # No-op when no actions
    item_no_actions = {"state": state}
    transform = _transforms.DeltaTCPPoseActions(enabled=True)
    assert transform(item_no_actions) is item_no_actions


def test_absolute_tcp_pose_actions_noop():
    """Test that AbsoluteTCPPoseActions is no-op when disabled or no actions."""
    state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.5])
    actions = np.array([[0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.8]])

    item = {"state": state, "actions": actions}

    # No-op when disabled
    transform = _transforms.AbsoluteTCPPoseActions(enabled=False)
    assert transform(item) is item

    # No-op when no actions
    item_no_actions = {"state": state}
    transform = _transforms.AbsoluteTCPPoseActions(enabled=True)
    assert transform(item_no_actions) is item_no_actions


def test_delta_tcp_pose_position_only():
    """Test that position delta is computed correctly (simple subtraction)."""
    # Use identity rotation to isolate position calculation
    state = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.5])  # identity axis-angle
    actions = np.array([
        [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.8],
        [7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0],
    ])

    item = {"state": state.copy(), "actions": actions.copy()}

    transform = _transforms.DeltaTCPPoseActions(pos_dims=3, rotation_repr="axis_angle")
    result = transform(item)

    # Position delta should be actions_pos - state_pos
    expected_delta_pos = actions[:, :3] - state[:3]
    assert np.allclose(result["actions"][:, :3], expected_delta_pos, atol=1e-5)

    # Gripper should be unchanged
    assert np.allclose(result["actions"][:, 6], actions[:, 6], atol=1e-5)


def test_invalid_rotation_repr():
    """Test that invalid rotation representation raises error."""
    with pytest.raises(ValueError, match="rotation_repr must be one of"):
        _transforms.DeltaTCPPoseActions(rotation_repr="invalid")

    with pytest.raises(ValueError, match="rotation_repr must be one of"):
        _transforms.AbsoluteTCPPoseActions(rotation_repr="invalid")


if __name__ == "__main__":
    print("=" * 60)
    print("Running DeltaTCPPoseActions and AbsoluteTCPPoseActions tests")
    print("=" * 60)

    tests = [
        ("test_delta_tcp_pose_actions_axis_angle", test_delta_tcp_pose_actions_axis_angle),
        ("test_delta_tcp_pose_actions_quaternion", test_delta_tcp_pose_actions_quaternion),
        ("test_delta_tcp_pose_actions_rotation_6d", test_delta_tcp_pose_actions_rotation_6d),
        ("test_delta_tcp_pose_actions_euler_xyz", test_delta_tcp_pose_actions_euler_xyz),
        ("test_rotation_repr_euler_alias", test_rotation_repr_euler_alias),
        ("test_delta_tcp_pose_actions_noop", test_delta_tcp_pose_actions_noop),
        ("test_absolute_tcp_pose_actions_noop", test_absolute_tcp_pose_actions_noop),
        ("test_delta_tcp_pose_position_only", test_delta_tcp_pose_position_only),
        ("test_invalid_rotation_repr", test_invalid_rotation_repr),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        exit(1)
