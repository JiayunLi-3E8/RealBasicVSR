import argparse
import glob
import os
import queue
import signal
import threading
import time
from rich.progress import Progress

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmedit.core import tensor2img

from realbasicvsr.models.builder import build_model

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script of RealBasicVSR')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=64,
        help='maximum sequence length to be processed(default=64).')
    parser.add_argument(
        '--split',
        type=int,
        default=1,
        help='the number of blocks to split the height and width of the image for processing(default=1).')
    parser.add_argument(
        '--save_as_png',
        action="store_true",
        help='save as png,if use this flag.')
    parser.add_argument(
        '--fps', type=float, default=25, help='FPS of the output video(default=25).')
    args = parser.parse_args()

    return args


def init_model(config, checkpoint=None):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.eval()

    return model


class InferenceProgerss:
    def __init__(self, imageTotal, split, maxSeqLen):
        self.__maxSeqLen = maxSeqLen
        self.__progress = Progress()
        self.__taskLoad = self.__progress.add_task(
            "[green]Preload: ", total=maxSeqLen)
        self.__taskSplit = self.__progress.add_task(
            "[yellow]Infer in split: ", total=split*split)
        self.__taskOutput = self.__progress.add_task(
            "[red]Output: ", total=maxSeqLen, visible=False)
        self.__taskInfer = self.__progress.add_task(
            "[cyan]Inferring...", total=imageTotal*split*split)
        self.__progress.start()

    def __del__(self):
        self.__progress.stop()

    def inferAdvance(self, advance):
        self.__progress.advance(self.__taskInfer, advance)

    def splitAdvance(self):
        self.__progress.advance(self.__taskSplit)

    def splitReset(self):
        self.__progress.reset(self.__taskSplit)

    def preloadAdvance(self):
        self.__progress.advance(self.__taskLoad)

    def preloadComplete(self):
        self.__progress.update(self.__taskLoad, completed=self.__maxSeqLen)

    def preloadReset(self):
        self.__progress.reset(self.__taskLoad, visible=True)

    def preloadHide(self):
        self.__progress.update(self.__taskLoad, visible=False)

    def outputAdvance(self):
        self.__progress.advance(self.__taskOutput)

    def outputReset(self, total=None):
        self.__progress.reset(self.__taskOutput, total=total, visible=True)

    def outputHide(self):
        self.__progress.update(self.__taskOutput, visible=False)


class OutputThead(threading.Thread):
    def __init__(self, outputDir, inferenceThead, inputHandles, inputIsVideo, fpsIfOutV=25, saveAsPng=False):
        threading.Thread.__init__(self)
        self.__outputQueue = queue.Queue()
        self.__keep = True
        self.__progress = None
        self.__inputHandles = inputHandles
        self.__inputIsVideo = inputIsVideo
        self.__saveAsPng = saveAsPng
        self.__outputDir = outputDir
        self.__outputIsVideo = os.path.splitext(
            outputDir)[1] in VIDEO_EXTENSIONS

        if self.__outputIsVideo:
            output_dir = os.path.dirname(outputDir)
            mmcv.mkdir_or_exist(output_dir)

            inputOne = inputHandles.read() if inputIsVideo else inputHandles[0]
            if inputIsVideo:
                imageOne = np.flip(inputOne, axis=2)
            else:
                imageOne = mmcv.imread(inputOne, channel_order='rgb')
            imageOne = torch.from_numpy(
                imageOne / 255.).permute(2, 0, 1).float()
            imageOne = imageOne.unsqueeze(0)
            inputs = torch.stack([imageOne], dim=1)
            h, w = inferenceThead.inference(inputs).shape[-2:]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.__video_writer = cv2.VideoWriter(
                outputDir, fourcc, fpsIfOutV, (w, h))
        else:
            mmcv.mkdir_or_exist(outputDir)
            self.__imageIndex = 0

    def putOutputs(self, outputs):
        self.__outputQueue.put(outputs)

    def run(self):
        while self.__keep:
            if self.__outputQueue.empty():
                time.sleep(0.1)
            else:
                outputs = self.__outputQueue.get()
                if self.__progress is not None:
                    self.__progress.outputReset(outputs.size(1))
                if self.__outputIsVideo:
                    for i in range(0, outputs.size(1)):
                        img = tensor2img(outputs[:, i, :, :, :])
                        self.__video_writer.write(img.astype(np.uint8))
                        if self.__progress is not None:
                            self.__progress.outputAdvance()
                    del img
                else:
                    for i in range(0, outputs.size(1)):
                        output = tensor2img(outputs[:, i, :, :, :])
                        if self.__inputIsVideo:
                            filename = f'frame{(self.__imageIndex+1):08}.jpg'
                        else:
                            filename = os.path.basename(
                                self.__inputHandles[self.__imageIndex])
                        self.__imageIndex += 1
                        if self.__saveAsPng:
                            file_extension = os.path.splitext(filename)[1]
                            filename = filename.replace(file_extension, '.png')
                        mmcv.imwrite(output, f'{self.__outputDir}/{filename}')
                        if self.__progress is not None:
                            self.__progress.outputAdvance()
                    del output
                self.__outputQueue.task_done()
                del outputs
                if self.__progress is not None:
                    self.__progress.outputHide()

    def setProgress(self, progress: InferenceProgerss):
        self.__progress = progress

    def end(self):
        self.__outputQueue.join()
        self.__keep = False
        if self.__outputIsVideo:
            cv2.destroyAllWindows()
            self.__video_writer.release()


class InferenceThead(threading.Thread):
    def __init__(self, model, split=1):
        threading.Thread.__init__(self)
        self.__inputs = []
        self.__ready = False
        self.__keep = True
        self.__split = split
        # map to cuda, if available
        self.__cuda_flag = torch.cuda.is_available()
        if self.__cuda_flag:
            self.__model = model.cuda()
        else:
            self.__model = model
        self.__outputThead = None
        self.__progress = None

    def setOutputThead(self, outputThead: OutputThead):
        self.__outputThead = outputThead

    def setProgress(self, progress: InferenceProgerss):
        self.__progress = progress

    def putInputs(self, inputs: list):
        if len(inputs) == 0:
            return
        for i, img in enumerate(inputs):
            img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
            inputs[i] = img.unsqueeze(0)
        inputs = torch.stack(inputs, dim=1)
        if self.__progress is not None:
            self.__progress.preloadComplete()

        while self.__ready:
            time.sleep(0.001)
        self.__inputs = inputs
        self.__ready = True

    def end(self):
        while self.__ready:
            time.sleep(0.001)
        self.__keep = False

    def run(self):
        while self.__keep:
            if self.__ready:
                outputs = self.inference(self.__inputs)
                self.__ready = False
                self.__outputThead.putOutputs(outputs)
                del outputs
            else:
                time.sleep(0.001)

    def inference(self, inputs: torch.Tensor):
        if len(inputs) < 1:
            return
        with torch.no_grad():
            if self.__progress is not None:
                self.__progress.splitReset()
            colBlockOutputs = []
            for colBlock in inputs.chunk(self.__split, 3):
                rowBlockOutputs = []
                for rowBlock in colBlock.chunk(self.__split, 4):
                    if self.__cuda_flag:
                        rowBlock = rowBlock.cuda()
                    outputs = self.__model(rowBlock, test_mode=True)[
                        'output'].cpu()
                    rowBlockOutputs.append(outputs)
                    if self.__progress is not None:
                        self.__progress.splitAdvance()
                        self.__progress.inferAdvance(outputs.size(1))
                colBlockOutputs.append(torch.cat(rowBlockOutputs, 4))
            return torch.cat(colBlockOutputs, 3)


def toExit(signalnum, frame):
    os._exit(0)


def main():
    signal.signal(signal.SIGINT, toExit)
    signal.signal(signal.SIGTERM, toExit)
    args = parse_args()

    file_extension = os.path.splitext(args.input_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:  # input is a video file
        inputHandles = mmcv.VideoReader(args.input_dir)
        isVideo = True
    elif file_extension == '':  # input is a directory
        inputHandles = sorted(glob.glob(f'{args.input_dir}/*'))
        isVideo = False
    else:
        raise ValueError('"input_dir" can only be a video or a directory.')

    # initialize the model
    model = init_model(args.config, args.checkpoint)

    inferenceThead = InferenceThead(model, args.split)
    outputThead = OutputThead(args.output_dir, inferenceThead,
                              inputHandles, isVideo, args.fps, args.save_as_png)
    inferenceThead.setOutputThead(outputThead)
    imageTotal = inputHandles.frame_cnt if isVideo else len(inputHandles)
    progress = InferenceProgerss(imageTotal, args.split, args.max_seq_len)
    inferenceThead.setProgress(progress)
    outputThead.setProgress(progress)
    inferenceThead.start()
    outputThead.start()
    inputs = []
    for inputHandle in inputHandles:
        if isVideo:
            inputs.append(np.flip(inputHandle, axis=2))
        else:
            inputs.append(mmcv.imread(inputHandle, channel_order='rgb'))
        progress.preloadAdvance()
        if len(inputs) >= args.max_seq_len:
            inferenceThead.putInputs(inputs)
            inputs = []
            progress.preloadReset()
    inferenceThead.putInputs(inputs)
    progress.preloadHide()
    inferenceThead.end()
    inferenceThead.join()
    outputThead.end()
    outputThead.join()


if __name__ == '__main__':
    main()
