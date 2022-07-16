#include <string>
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace std;

int main() {
    cout << "\x7a\u00df\u6c34\U0001f34c" << endl;
    {
        cout << "char: ";
        string s = "\x7a\u00df\u6c34\U0001f34c";
        for (size_t i = 0; i < s.size(); i++) {
            cout << setw(2) << setfill('0') << hex;
            cout << hex << (uint32_t)(uint8_t)s[i] << ' ';
        }
        cout << endl;
    }
    {
        cout << "wchar_t: ";
        wstring s = L"\x7a\u00df\u6c34\U0001f34c";
        for (size_t i = 0; i < s.size(); i++) {
            cout << setw(sizeof(wchar_t) * 2) << setfill('0') << hex;
            cout << (uint32_t)s[i] << ' ';
        }
        cout << endl;
    }
    //{
        //cout << "char8_t: ";
        //u8string s = u8"\x7a\u00df\u6c34\U0001f34c";
        //for (size_t i = 0; i < s.size(); i++) {
            //cout << setw(2) << setfill('0') << hex;
            //cout << (uint16_t)s[i] << ' ';
        //}
        //cout << endl;
    //}
    {
        cout << "char16_t: ";
        u16string s = u"\x7a\u00df\u6c34\U0001f34c";
        for (size_t i = 0; i < s.size(); i++) {
            cout << setw(4) << setfill('0') << hex;
            cout << (uint16_t)s[i] << ' ';
        }
        cout << endl;
    }
    {
        cout << "char32_t: ";
        u32string s = U"\x7a\u00df\u6c34\U0001f34c";
        for (size_t i = 0; i < s.size(); i++) {
            cout << setw(8) << setfill('0') << hex;
            cout << (uint32_t)s[i] << ' ';
        }
        cout << endl;
    }
}
