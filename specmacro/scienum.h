#pragma once
#include <string>
namespace scienum
{
	namespace details
	{
		template <bool Cond>
		struct my_enable_if{
		};
		template <>
		struct my_enable_if<true>{
			typedef std::string type;
		};

		template <typename T, T N>
		std::string get_enum_name_static(){
#if defined(_MSC_VER)
			std::string s = __FUNCSIG__;
			size_t pos = s.find_last_of(',');
			pos += 1;
			size_t pos2 = s.find('>', pos);
#else
			std::string s = __PRETTY_FUNCTION__;
			size_t pos = s.find("N = ");
			pos += 4;
			size_t pos2 = s.find_first_of(";]", pos);
#endif
			s = s.substr(pos, pos2 - pos);
			size_t pos3 = s.find("::");
			if (pos3 != s.npos)
				s = s.substr(pos3 + 2);
			return s;
		}

	}
	template <typename T, int Beg, int End>
	typename details::my_enable_if<Beg == End>::type get_enum_name(T v){
		return "";
	}
	template <typename T, int Beg, int End>
	typename details::my_enable_if<Beg != End>::type get_enum_name(T v){
		if (Beg == v)
			return details::get_enum_name_static<T, (T)Beg>();
		return get_enum_name<T, T(Beg + 1), End>(v);
	}
	template <typename T>
	std::string get_enum_name(T v){
		return get_enum_name<T, 0, 256>(v);
	}

	template <class T, T Beg, T End>
	T enum_from_name(std::string const &s) {
		for (int i = (int)Beg; i < (int)End; i++) {
			if (s == get_enum_name<T,Beg,End>((T)i)) {
				return (T)i;
			}
		}
		throw;
	}

	template <class T>
	T enum_from_name(std::string const &s) {
		return enum_from_name<T, (T)0, (T)256>(s);
	}

}
