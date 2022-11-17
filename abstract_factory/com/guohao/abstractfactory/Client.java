package abstract_factory.com.guohao.abstractfactory;

import abstract_factory.com.guohao.abstractfactory.aliyun.AliyunOssFactory;
import abstract_factory.com.guohao.abstractfactory.factory.AbstractOssFactory;
import abstract_factory.com.guohao.abstractfactory.factory.product.OssImage;
import abstract_factory.com.guohao.abstractfactory.factory.product.OssVideo;

public class Client {
    public static void main(String[] args) {
//        AbstractOssFactory factory = new QiniuOssFactory();
        AbstractOssFactory factory = new AliyunOssFactory();

        OssImage ossImage = factory.uploadImage(new byte[1024]);
        OssVideo ossVideo = factory.uploadVideo(new byte[1024]);
        System.out.println(ossImage.getThumb());
        System.out.println(ossImage.getWatermark());
        System.out.println(ossImage.getEnhance());
        System.out.println(ossVideo.get720P());
        System.out.println(ossVideo.get1080p());
    }
}
