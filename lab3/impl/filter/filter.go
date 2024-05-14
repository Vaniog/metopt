package filter

func Map[T any, U any](ts []T, mapFunc func(T) U) (us []U) {
	for i := range ts {
		us = append(us, mapFunc(ts[i]))
	}
	return
}
