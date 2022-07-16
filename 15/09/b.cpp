#include <string>
#include <string_view>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <locale>

using namespace std;

template <class Internal = wchar_t>
string convcodec(string_view src, locale const &srcLoc, locale const &dstLoc) {
    auto strCodecvtError = [] (codecvt_base::result res) -> string {
        if (res == codecvt_base::result::noconv)
            return "noconv";
        if (res == codecvt_base::result::error)
            return "error";
        if (res == codecvt_base::result::partial)
            return "partial";
        return "ok";
    };

    string dst;
    basic_string<Internal> tmp;
    mbstate_t mbstate = {};

    auto &srcFacet = use_facet<codecvt<Internal, char, mbstate_t>>(srcLoc);
    tmp.resize(src.size());

    const char *src_begin = src.data();
    const char *src_end = src.data() + src.size();
    const char *src_next;
    Internal *tmp_begin = tmp.data();
    Internal *tmp_end = tmp.data() + tmp.size();
    Internal *tmp_next;
    auto resIn = srcFacet.in(mbstate, src_begin, src_end, src_next, tmp_begin, tmp_end, tmp_next);
    if (resIn != codecvt_base::result::ok)
        throw runtime_error("cannot decode input sequence: " + strCodecvtError(resIn));
    size_t tmp_size = tmp_next - tmp_begin;
    tmp_end = tmp_next;

    auto &dstFacet = use_facet<codecvt<Internal, char, mbstate_t>>(dstLoc);
    dst.resize(tmp_size * dstFacet.max_length());

    char *dst_begin = dst.data();
    char *dst_end = dst.data() + dst.size();
    char *dst_next;
    const Internal *tmp_next_next;
    auto resOut = dstFacet.out(mbstate, tmp_begin, tmp_end, tmp_next_next, dst_begin, dst_end, dst_next);
    if (resOut != codecvt_base::result::ok)
        throw runtime_error("cannot encode output sequence: " + strCodecvtError(resOut));
    size_t dst_size = dst_next - dst_begin;

    dst.resize(dst_size);
    return dst;
}

void showstr(string_view s) {
    for (char c: s) {
        cout << hex << setfill('0') << setw(2);
        cout << (uint32_t)(uint8_t)c << ' ';
    }
    cout << s << endl;
}

int main() {
    string s = "你好";
    //string s = "你好\U0001f34c";  // error: cannot encode U+1F34C
    showstr(s);
    s = convcodec(s, locale("en_US.UTF-8"), locale("zh_CN.GBK"));
    showstr(s);
}
