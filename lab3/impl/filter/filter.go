package filter

func Map[T any, U any](ts []T, mapFunc func(T) U) (us []U) {
	for i := range ts {
		us = append(us, mapFunc(ts[i]))
	}
	return
}

func MapWithError[T any, U any](ts []T, mapFunc func(T) (U, error)) (us []U, err error) {
	var res U
	for i := range ts {
		res, err = mapFunc(ts[i])
		if err != nil {
			return
		}
		us = append(us, res)
	}
	return
}
