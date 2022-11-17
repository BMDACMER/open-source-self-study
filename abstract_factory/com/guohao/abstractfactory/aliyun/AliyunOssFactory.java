package abstract_factory.com.guohao.abstractfactory.aliyun;

import abstract_factory.com.guohao.abstractfactory.aliyun.product.AliyunOssImage;
import abstract_factory.com.guohao.abstractfactory.aliyun.product.AliyunOssVideo;
import abstract_factory.com.guohao.abstractfactory.factory.AbstractOssFactory;
import abstract_factory.com.guohao.abstractfactory.factory.product.OssImage;
import abstract_factory.com.guohao.abstractfactory.factory.product.OssVideo;

public class AliyunOssFactory implements AbstractOssFactory {
    @Override
    public OssImage uploadImage(byte[] bytes) {
        return new AliyunOssImage(bytes, "guohao", true);
    }

    @Override
    public OssVideo uploadVideo(byte[] bytes) {
        return new AliyunOssVideo(bytes, "guohao");
    }
}
