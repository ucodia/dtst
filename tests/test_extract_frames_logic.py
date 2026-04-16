import shutil

from dtst.core.extract_frames import (
    _FFMPEG_TIME_RE,
    _build_ffmpeg_cmd,
    _check_ffmpeg,
)


class TestFfmpegTimeRegex:
    def test_matches_basic_out_time_us(self):
        m = _FFMPEG_TIME_RE.search("out_time_us=12345")
        assert m is not None
        assert m.group(1) == "12345"

    def test_captures_digits_only(self):
        m = _FFMPEG_TIME_RE.search("out_time_us=987654321")
        assert m is not None
        assert m.group(1) == "987654321"

    def test_matches_within_larger_string(self):
        line = "frame=42\nout_time_us=555000\nspeed=1.2x"
        m = _FFMPEG_TIME_RE.search(line)
        assert m is not None
        assert m.group(1) == "555000"

    def test_matches_zero_value(self):
        m = _FFMPEG_TIME_RE.search("out_time_us=0")
        assert m is not None
        assert m.group(1) == "0"

    def test_no_match_in_unrelated_line(self):
        assert _FFMPEG_TIME_RE.search("frame=100") is None

    def test_no_match_on_similar_keys(self):
        assert _FFMPEG_TIME_RE.search("out_time=00:00:01.000000") is None

    def test_no_match_on_empty_string(self):
        assert _FFMPEG_TIME_RE.search("") is None

    def test_no_match_when_value_is_non_numeric(self):
        assert _FFMPEG_TIME_RE.search("out_time_us=N/A") is None


class TestCheckFfmpeg:
    def test_returns_false_when_which_returns_none(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        assert _check_ffmpeg() is False

    def test_returns_true_when_which_returns_path(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/ffmpeg")
        assert _check_ffmpeg() is True

    def test_returns_true_when_which_returns_empty_string_is_false(self, monkeypatch):
        # shutil.which returning "" would be a truthy path string only if non-empty;
        # empty string is falsy but `is not None` makes it True. Document behavior.
        monkeypatch.setattr(shutil, "which", lambda name: "")
        assert _check_ffmpeg() is True

    def test_queries_ffmpeg_binary_name(self, monkeypatch):
        seen: list[str] = []

        def fake_which(name):
            seen.append(name)
            return "/bin/ffmpeg"

        monkeypatch.setattr(shutil, "which", fake_which)
        _check_ffmpeg()
        assert seen == ["ffmpeg"]


class TestBuildFfmpegCmd:
    def test_returns_list_starting_with_ffmpeg(self):
        cmd = _build_ffmpeg_cmd("in.mp4", "out_%04d.jpg", 10.0, "jpg")
        assert isinstance(cmd, list)
        assert cmd[0] == "ffmpeg"

    def test_contains_input_flag_and_video_path(self):
        cmd = _build_ffmpeg_cmd("/videos/clip.mp4", "out_%04d.jpg", 5.0, "jpg")
        idx = cmd.index("-i")
        assert cmd[idx + 1] == "/videos/clip.mp4"

    def test_output_pattern_is_last_element(self):
        pattern = "/tmp/frames/clip_%04d.jpg"
        cmd = _build_ffmpeg_cmd("in.mp4", pattern, 10.0, "jpg")
        assert cmd[-1] == pattern

    def test_qv_is_2_for_jpg(self):
        cmd = _build_ffmpeg_cmd("in.mp4", "out_%04d.jpg", 10.0, "jpg")
        idx = cmd.index("-q:v")
        assert cmd[idx + 1] == "2"

    def test_qv_is_0_for_png(self):
        cmd = _build_ffmpeg_cmd("in.mp4", "out_%04d.png", 10.0, "png")
        idx = cmd.index("-q:v")
        assert cmd[idx + 1] == "0"

    def test_qv_is_0_for_any_non_jpg(self):
        cmd = _build_ffmpeg_cmd("in.mp4", "out_%04d.webp", 10.0, "webp")
        idx = cmd.index("-q:v")
        assert cmd[idx + 1] == "0"

    def test_select_expr_embeds_keyframes_interval(self):
        cmd = _build_ffmpeg_cmd("in.mp4", "out_%04d.jpg", 7.5, "jpg")
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        assert "7.5" in vf_arg
        assert "select=" in vf_arg

    def test_select_expr_embeds_integer_interval(self):
        cmd = _build_ffmpeg_cmd("in.mp4", "out_%04d.jpg", 10.0, "jpg")
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        assert "10.0" in vf_arg

    def test_includes_skip_frame_nokey(self):
        cmd = _build_ffmpeg_cmd("in.mp4", "out_%04d.jpg", 10.0, "jpg")
        idx = cmd.index("-skip_frame")
        assert cmd[idx + 1] == "nokey"

    def test_includes_progress_pipe(self):
        cmd = _build_ffmpeg_cmd("in.mp4", "out_%04d.jpg", 10.0, "jpg")
        idx = cmd.index("-progress")
        assert cmd[idx + 1] == "pipe:1"

    def test_all_elements_are_strings(self):
        cmd = _build_ffmpeg_cmd("in.mp4", "out_%04d.jpg", 10.0, "jpg")
        assert all(isinstance(arg, str) for arg in cmd)
