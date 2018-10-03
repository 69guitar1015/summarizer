from contextlib import contextmanager
import chainer
from tensorboardX import SummaryWriter


@contextmanager
def enable(summarizer=None, global_step=None):
    if summarizer is None:
        summarizer = _summarizer

    assert summarizer.writer is not None, 'Call initialize_writer(log_dir) first'

    summarizer.report = True
    summarizer.global_step = global_step

    yield

    summarizer.report = False
    summarizer.global_step = None


class Summarizer(object):
    def __init__(self):
        self.report = False
        self.global_step = None
        self.writer = None

    def initialize_writer(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_scalar(
            tag, scalar_value, global_step=global_step, walltime=walltime)

    def add_scalars(self,
                    main_tag,
                    tag_scalar_dict,
                    global_step=None,
                    walltime=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_scalars(
            self,
            main_tag,
            tag_scalar_dict,
            global_step=global_step,
            walltime=walltime)

    def add_histogram(self,
                      tag,
                      values,
                      global_step=None,
                      bins='tensorflow',
                      walltime=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        if isinstance(values, chainer.cuda.cupy.ndarray):
            values = chainer.cuda.to_cpu(values)

        self.writer.add_histogram(
            tag, values, global_step=global_step, bins=bins, walltime=walltime)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_image(
            tag, img_tensor, global_step=global_step, walltime=walltime)

    def add_image_with_boxes(self,
                             tag,
                             img_tensor,
                             box_tensor,
                             global_step=None,
                             walltime=None,
                             **kwargs):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_image_with_boxes(
            tag,
            img_tensor,
            box_tensor,
            global_step=global_step,
            walltime=walltime,
            **kwargs)

    def add_figure(self,
                   tag,
                   figure,
                   global_step=None,
                   close=True,
                   walltime=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_figure(
            tag,
            figure,
            global_step=global_step,
            close=close,
            walltime=walltime)

    def add_video(self,
                  tag,
                  vid_tensor,
                  global_step=None,
                  fps=4,
                  walltime=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_video(
            tag,
            vid_tensor,
            global_step=global_step,
            fps=fps,
            walltime=walltime)

    def add_audio(self,
                  tag,
                  snd_tensor,
                  global_step=None,
                  sample_rate=44100,
                  walltime=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_audio(
            tag,
            snd_tensor,
            global_step=global_step,
            sample_rate=sample_rate,
            walltime=walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_text(
            tag, text_string, global_step=global_step, walltime=walltime)

    def add_graph_onnx(self, prototxt):
        if not self.report:
            return

        self.writer.add_graph_onnx(self, prototxt)

    def add_graph(self, model, input_to_model=None, verbose=False, **kwargs):
        if not self.report:
            return

        self.writer.add_graph(
            model, input_to_model=input_to_model, verbose=verbose, **kwargs)

    def add_embedding(self,
                      mat,
                      metadata=None,
                      label_img=None,
                      global_step=None,
                      tag='default',
                      metadata_header=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_embedding(
            mat,
            metadata=metadata,
            label_img=label_img,
            global_step=global_step,
            tag=tag,
            metadata_header=metadata_header)

    def add_pr_curve(self,
                     tag,
                     labels,
                     predictions,
                     global_step=None,
                     num_thresholds=127,
                     weights=None,
                     walltime=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_pr_curve(
            tag,
            labels,
            predictions,
            global_step=global_step,
            num_thresholds=num_thresholds,
            weights=weights,
            walltime=walltime)

    def add_pr_curve_raw(self,
                         tag,
                         true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         global_step=None,
                         num_thresholds=127,
                         weights=None,
                         walltime=None):
        if not self.report:
            return

        if global_step is None and self.global_step is not None:
            global_step = self.global_step

        self.writer.add_pr_curve_raw(
            tag,
            true_positive_counts,
            false_positive_counts,
            true_negative_counts,
            false_negative_counts,
            precision,
            recall,
            global_step=global_step,
            num_thresholds=num_thresholds,
            weights=weights,
            walltime=walltime)

    def add_custom_scalars_multilinechart(self,
                                          tags,
                                          category='default',
                                          title='untitled'):
        if not self.report:
            return
        self.writer.add_custom_scalars_multilinechart(
            tags, category=category, title=title)

    def add_custom_scalars_marginchart(self,
                                       tags,
                                       category='default',
                                       title='untitled'):
        if not self.report:
            return
        self.writer.add_custom_scalars_marginchart(
            tags, category=category, title=title)

    def add_custom_scalars(self, layout):
        if not self.report:
            return
        self.writer.add_custom_scalars(layout)


# Default summarizer
_summarizer = Summarizer()
initialize_writer = _summarizer.initialize_writer
add_scalar = _summarizer.add_scalar
add_scalars = _summarizer.add_scalars
add_histogram = _summarizer.add_histogram
add_image = _summarizer.add_image
add_image_with_boxes = _summarizer.add_image_with_boxes
add_figure = _summarizer.add_figure
add_video = _summarizer.add_video
add_audio = _summarizer.add_audio
add_text = _summarizer.add_text
add_graph_onnx = _summarizer.add_graph_onnx
add_graph = _summarizer.add_graph
add_embedding = _summarizer.add_embedding
add_pr_curve = _summarizer.add_pr_curve
add_pr_curve_raw = _summarizer.add_pr_curve_raw
add_custom_scalars_multilinechart = _summarizer.add_custom_scalars_multilinechart
add_custom_scalars_marginchart = _summarizer.add_custom_scalars_marginchart
add_custom_scalars = _summarizer.add_custom_scalars
