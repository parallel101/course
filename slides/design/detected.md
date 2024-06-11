```cpp
struct API {
    bool is_device_ready(std::string dev_name);
    void play_device(std::string dev_name);
    void init_device(std::string dev_name);
};

if (api->is_device_ready("CD"))
    api->play_device("CD");
else
    api->init_device("CD");
```

```cpp
struct Device {
    virtual bool is_ready() = 0;
    virtual void play() = 0;
    virtual void init() = 0;
};

struct API {
    Device *get_device(std::string dev_name);
};

Device *dev = api->get_device("CD");
if (dev->is_ready())
    dev->play();
else
    dev->init();
```
