import argparse

import cv2
import datetime


def main(args):
    if args.src_path:
        # video_path = 'rtsp://admin:12345678a@{}:554/h264/ch1/main/av_stream'.format(args.src_path)
        video_path = 'rtsp://admin:qdby1234@{}:554/Streaming/Channels/1'.format(args.src_path)
        print(video_path)
        vid = cv2.VideoCapture(video_path)
    else:
        vid = cv2.VideoCapture(0)

    now_time = datetime.datetime.now()
    time_str = now_time.strftime('%Y-%m-%d %H:%M:%S,%f')[:23]

    if args.validate:
        img_path = 'image_{}.jpg'
    else:
        img_path = ''

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = vid.get(cv2.CAP_PROP_FPS)
    size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('video_{}.avi'.format(time_str), fourcc, fps, size)

    cnt = 0
    print('start')
    while True:
        ret, frame = vid.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            out.write(frame)
            if cnt == 0 and args.validate:
                cv2.imwrite(frame, img_path)

            cnt += 1
            if cnt % 100 == 0:
                print(cnt)

            if args.show:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            raise ValueError("No image!")

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--src-path', type=str, default='10.83.172.78', help='src path')
    parser.add_argument('-V', '--validate', action='store_true', default=True, help='validate flag')
    parser.add_argument('--show', action='store_true', default=False, help='show frame')
    opt = parser.parse_args()
    main(opt)
