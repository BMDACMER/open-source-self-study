package abstract_factory.com.guohao.abstractfactory.qiniu;

import abstract_factory.com.guohao.abstractfactory.factory.AbstractOssFactory;
import abstract_factory.com.guohao.abstractfactory.factory.product.OssImage;
import abstract_factory.com.guohao.abstractfactory.factory.product.OssVideo;
import abstract_factory.com.guohao.abstractfactory.qiniu.product.QiniuOssImage;
import abstract_factory.com.guohao.abstractfactory.qiniu.product.QiniuOssVideo;

public class QiniuOssFactory implements AbstractOssFactory {
    @Override
    public OssImage uploadImage(byte[] bytes) {
        return new QiniuOssImage(bytes, "guohao");
    }

    @Override
    public OssVideo uploadVideo(byte[] bytes) {
        return new QiniuOssVideo(bytes, "guohao");
    }
}
