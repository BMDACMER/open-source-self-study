package abstract_factory.com.guohao.abstractfactory.factory;

import abstract_factory.com.guohao.abstractfactory.factory.product.OssImage;
import abstract_factory.com.guohao.abstractfactory.factory.product.OssVideo;

public interface AbstractOssFactory {
    public OssImage uploadImage(byte[] bytes);
    public OssVideo uploadVideo(byte[] bytes);
}
