## Setup Android SDK

* From commandline

```bash
mkdir android_sdk
cd android_sdk

# download commandline tools from https://developer.android.com/studio#command-tools

wget https://dl.google.com/android/repository/commandlinetools-linux-8512546_latest.zip

unzip commandlinetools-linux-8512546_latest.zip

# install android platform
cmdline-tools/bin/sdkmanager --install --sdk_root=. "platforms;android-30"

# install build tools
cmdline-tools/bin/sdkmanager --install --sdk_root=. "build-tools;30.0.3"

# install ndk
cmdline-tools/bin/sdkmanager --install --sdk_root=. "ndk;21.1.6352462"
```

* Android studio

You can also setup your Android SDK with Android Studio, if you have a machine with GUI, see [Android Studio documents](https://developer.android.com/studio/install) for how to use it.

Note: We refer to [Pytorch android StreamingASR demo](https://github.com/pytorch/android-demo-app/tree/master/StreamingASR) to setup this android demo, the platform and build-tools version above are known to work. You might encounter issues if you use other versions.

## Build apk

* Get sherpa

```bash
git clone https://github.com/k2-fsa/sherpa.git
```

* Prepare model and vocabulary file

Put `jit.pt` and `tokens.txt` into `device/android/app/src/main/assets`.
See [device/android/app/src/main/assets/README.md](./app/src/main/assets/README.md) for more details.

* Build with gradle

```
cd device/android/app

./gradlew build
```

Now, you can get apk from `app/build/outputs/apk/app-release.apk`.

